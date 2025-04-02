import google.generativeai as genai
import sounddevice as sd
import numpy as np
import threading
import queue
import os
import sys
import curses
import time
import logging

# --- Configuration ---
# Audio settings (Input - Microphone)
INPUT_SAMPLE_RATE = 16000
INPUT_CHANNELS = 1
INPUT_DTYPE = np.int16  # 16-bit PCM
INPUT_BLOCK_DURATION_MS = 50 # Process audio in chunks (milliseconds)
INPUT_BLOCK_SIZE = int(INPUT_SAMPLE_RATE * INPUT_BLOCK_DURATION_MS / 1000)

# Audio settings (Output - Speaker)
OUTPUT_SAMPLE_RATE = 24000 # Gemini output spec
OUTPUT_CHANNELS = 1
OUTPUT_DTYPE = np.int16

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "models/gemini-1.5-flash" # Or another model supporting stream_audio

# UI Update Interval
UI_UPDATE_INTERVAL = 0.1 # seconds

# Logging Setup (optional, logs to a file)
logging.basicConfig(filename='gemini_audio.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

# --- Global State and Communication ---
input_audio_queue = queue.Queue()
output_audio_queue = queue.Queue()
stop_event = threading.Event()
app_status = "INITIALIZING"
status_lock = threading.Lock()
last_error = None

# --- Audio Input ---
def audio_input_callback(indata, frames, time, status):
    """Callback function for sounddevice input stream."""
    # Declare global intention at the beginning
    global app_status, last_error

    if status:
        logging.warning(f"Input audio status: {status}")
        with status_lock:
            # Now assignment is fine after global declaration
            app_status = "ERROR"
            last_error = f"Audio Input Error: {status}"
    if not stop_event.is_set():
        # Convert NumPy array to bytes and add to queue
        input_audio_queue.put(indata.tobytes())

def input_audio_thread_func():
    """Thread function to manage the audio input stream."""
    # Declare global intention at the beginning
    global app_status, last_error
    logging.info("Input audio thread started.")
    try:
        with sd.InputStream(
            samplerate=INPUT_SAMPLE_RATE,
            channels=INPUT_CHANNELS,
            dtype=INPUT_DTYPE,
            blocksize=INPUT_BLOCK_SIZE, # Process in chunks
            callback=audio_input_callback
        ):
            logging.info("Audio input stream started.")
            with status_lock:
                 # Assignment fine after global declaration
                app_status = "LISTENING"
            stop_event.wait() # Keep stream open until stop_event is set
    except Exception as e:
        logging.exception("Error in input audio thread:")
        with status_lock:
             # Assignment fine after global declaration
            app_status = "ERROR"
            last_error = f"Input Thread Failed: {e}"
        stop_event.set() # Signal other threads to stop on critical error
    finally:
        logging.info("Input audio thread finished.")

# --- Audio Output ---
def output_audio_thread_func():
    """Thread function to manage the audio output stream."""
     # Declare global intention at the beginning
    global app_status, last_error
    logging.info("Output audio thread started.")
    try:
        with sd.OutputStream(
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=OUTPUT_CHANNELS,
            dtype=OUTPUT_DTYPE
        ) as stream:
            logging.info("Audio output stream started.")
            while not stop_event.is_set():
                try:
                    # Get audio data from queue, with timeout
                    audio_data_bytes = output_audio_queue.get(timeout=0.1)
                    if audio_data_bytes:
                        with status_lock:
                             # Assignment fine after global declaration
                            app_status = "SPEAKING"
                        # Convert bytes back to NumPy array for playback
                        audio_data_np = np.frombuffer(audio_data_bytes, dtype=OUTPUT_DTYPE)
                        stream.write(audio_data_np)
                        output_audio_queue.task_done() # Signal task completion
                    else:
                        # Received None, likely a signal to stop or flush?
                         logging.debug("Received None in output queue.")

                except queue.Empty:
                    # No data, check status and continue
                     with status_lock:
                        # Assignment fine after global declaration
                        # If not speaking, go back to listening (or error state if already error)
                        if app_status == "SPEAKING":
                            app_status = "LISTENING"
                     continue
                except Exception as e:
                     logging.exception("Error during audio playback:")
                     with status_lock:
                         # Assignment fine after global declaration
                         app_status = "ERROR"
                         last_error = f"Playback Error: {e}"
                     # Don't stop the whole app for playback errors? Optional.
                     # stop_event.set()

    except Exception as e:
        logging.exception("Error initializing output audio stream:")
        with status_lock:
            # Assignment fine after global declaration
            app_status = "ERROR"
            last_error = f"Output Thread Failed: {e}"
        stop_event.set() # Signal other threads to stop
    finally:
        logging.info("Output audio thread finished.")
        # Ensure queue is cleared if stopping abnormally?
        while not output_audio_queue.empty():
            try: output_audio_queue.get_nowait()
            except queue.Empty: break
            output_audio_queue.task_done()


# --- Gemini Interaction ---
def audio_chunk_generator():
    """Generator that yields audio chunks from the input queue."""
    while not stop_event.is_set():
        try:
            chunk = input_audio_queue.get(timeout=0.1)
            yield chunk
            input_audio_queue.task_done()
        except queue.Empty:
             continue
        except Exception as e:
             logging.exception("Error getting chunk from input queue for generator:")
             stop_event.set()
             break
    logging.info("Audio chunk generator finished.")


def gemini_audio_thread_func():
    """Thread function to handle communication with the Gemini API."""
    # Declare global intention at the beginning
    global app_status, last_error
    logging.info("Gemini API thread started.")

    if not GEMINI_API_KEY:
        logging.error("GOOGLE_API_KEY environment variable not set.")
        with status_lock:
             # Assignment fine after global declaration
            app_status = "ERROR"
            last_error = "API Key Not Found"
        stop_event.set()
        return

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)

        audio_stream = genai.types.AudioStream(
            stream=audio_chunk_generator(),
            sample_rate=INPUT_SAMPLE_RATE,
            channels=INPUT_CHANNELS
        )
        output_audio_spec = genai.types.OutputAudioSpec(
            sample_rate=OUTPUT_SAMPLE_RATE,
            encoding='linear16'
        )
        system_instruction = "You are a helpful AI assistant. Respond concisely."
        request = genai.types.GenerateContentRequest(
            contents=[genai.types.Content(parts=[genai.types.Part()])],
            audio_stream=audio_stream,
            output_audio_spec=output_audio_spec,
            system_instruction=system_instruction,
        )

        logging.info("Starting stream_audio request to Gemini...")
        with status_lock:
             # Assignment fine after global declaration
            if app_status != "ERROR": app_status = "LISTENING"

        response_stream = model.stream_audio(request=request)

        for response in response_stream:
            if stop_event.is_set():
                logging.info("Stop event set, breaking Gemini response loop.")
                break

            if response.audio_data:
                logging.debug(f"Received audio data chunk: {len(response.audio_data)} bytes")
                output_audio_queue.put(response.audio_data)
            else:
                 logging.debug(f"Received non-audio response part: {response}")
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                      logging.error(f"Gemini blocked request: {response.prompt_feedback.block_reason_message}")
                      with status_lock:
                         # Assignment fine after global declaration
                         app_status = "ERROR"
                         last_error = f"Gemini Blocked: {response.prompt_feedback.block_reason_message}"
                      stop_event.set()
                      break

        logging.info("Finished processing Gemini response stream.")

    except Exception as e:
        logging.exception("Error in Gemini API thread:")
        with status_lock:
            # Assignment fine after global declaration
            app_status = "ERROR"
            last_error = f"Gemini API Error: {e}"
        stop_event.set()
    finally:
        logging.info("Gemini API thread finished.")


# --- Curses UI ---
def draw_ui(stdscr):
    """Draws the full-screen UI using curses."""
    stdscr.nodelay(True)
    stdscr.timeout(int(UI_UPDATE_INTERVAL * 1000))
    curses.curs_set(0)

    # Initialize colors (check if terminal supports color)
    has_color = curses.has_colors()
    if has_color:
        curses.start_color()
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)    # Error
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Listening
        curses.init_pair(3, curses.COLOR_BLUE, curses.COLOR_BLACK)   # Speaking
        default_color = curses.color_pair(0)
        error_color = curses.color_pair(1)
        listen_color = curses.color_pair(2)
        speak_color = curses.color_pair(3)
    else:
        # Define fallbacks if no color
        default_color = curses.A_NORMAL
        error_color = curses.A_BOLD
        listen_color = curses.A_NORMAL
        speak_color = curses.A_BOLD


    k = 0
    while k != ord('q') and not stop_event.is_set():
        try:
            stdscr.erase()
            height, width = stdscr.getmaxyx()

            # --- Get current status safely ---
            with status_lock:
                current_status = app_status
                current_error = last_error

            # --- Title ---
            title = "Gemini Live Audio Assistant"
            stdscr.attron(curses.A_BOLD)
            title_x = max(0, (width - len(title)) // 2)
            stdscr.addstr(0, title_x, title)
            stdscr.attroff(curses.A_BOLD)

            # --- Status Display ---
            status_line = 2
            status_text = f"Status: {current_status}"
            color = default_color
            if current_status == "LISTENING":
                color = listen_color
            elif current_status == "SPEAKING":
                color = speak_color
            elif current_status == "ERROR":
                color = error_color
                status_text += f" - Error: {current_error}" if current_error else ""

            stdscr.addstr(status_line, 1, status_text[:width-2], color) # Truncate if too long

             # --- Input/Output Queue Info (Optional Debug) ---
            debug_line = 4
            in_q_size = input_audio_queue.qsize()
            out_q_size = output_audio_queue.qsize()
            debug_text = f"Input Q: {in_q_size:<5} Output Q: {out_q_size:<5}"
            stdscr.addstr(debug_line, 1, debug_text[:width-2])


            # --- Instructions ---
            instruction_line = height - 2
            instruction = "Press 'q' to quit"
            inst_x = max(0, (width - len(instruction)) // 2)
            stdscr.addstr(instruction_line, inst_x, instruction)

            # --- Border ---
            stdscr.border(0)

            stdscr.refresh()

            # --- Check for input ---
            k = stdscr.getch()

        except curses.error as e:
             # Handle screen too small error gracefully
             if "addwstr" in str(e) or "addstr" in str(e):
                 logging.warning(f"Curses error likely due to small screen: {e}")
                 # Optionally display a minimal message if possible
                 try:
                     stdscr.erase()
                     stdscr.addstr(0, 0, "Screen too small. Press 'q'.")
                     stdscr.refresh()
                     time.sleep(1) # Give user time to see
                 except: pass # Ignore errors during error display
                 # Keep checking for 'q' even if drawing fails
                 while k != ord('q'):
                      k = stdscr.getch()
                      time.sleep(0.1)

             else:
                 logging.exception("Unhandled Curses UI error:")
                 stop_event.set() # Stop app on other UI errors
             break # Exit UI loop on error
        except Exception as e:
             logging.exception("Error during curses UI update:")
             stop_event.set()
             break


    # --- Exit ---
    stop_event.set() # Ensure stop is signalled if 'q' was pressed or loop broken

def main_ui_wrapper(stdscr):
    """Wrapper for curses application setup and teardown."""
    draw_ui(stdscr)

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Gemini Live Audio App...")
    print(f"Input: {INPUT_SAMPLE_RATE}Hz, Output: {OUTPUT_SAMPLE_RATE}Hz")
    print("Check gemini_audio.log for detailed logs.")

    # --- Create and start threads ---
    threads = []
    try:
        input_thread = threading.Thread(target=input_audio_thread_func, name="AudioInputThread", daemon=True)
        output_thread = threading.Thread(target=output_audio_thread_func, name="AudioOutputThread", daemon=True)
        gemini_thread = threading.Thread(target=gemini_audio_thread_func, name="GeminiAPIThread", daemon=True)

        threads.extend([input_thread, output_thread, gemini_thread])

        input_thread.start()
        output_thread.start()
        gemini_thread.start()

        time.sleep(0.5)
        curses.wrapper(main_ui_wrapper)

    except Exception as e:
        logging.exception("Fatal error during app startup or UI execution:")
        print(f"\nApplication failed: {e}")
        stop_event.set()
    finally:
        print("\nShutting down...")
        stop_event.set()

        for thread in threads:
            try:
                 thread.join(timeout=2.0)
                 if thread.is_alive():
                     print(f"Warning: Thread {thread.name} did not finish cleanly.")
            except Exception as e:
                 print(f"Error joining thread {thread.name}: {e}")

        print("Application finished.")