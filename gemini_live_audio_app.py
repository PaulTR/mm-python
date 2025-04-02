import google.generativeai as genai
import google.generativeai.types as types # Import types
import sounddevice as sd
import numpy as np
import threading
import queue
import os
import sys
import curses
import time
import logging
import asyncio # Import asyncio

# --- Configuration ---
# Audio settings (Input - Microphone)
INPUT_SAMPLE_RATE = 16000 # Required by Live API
INPUT_CHANNELS = 1
INPUT_DTYPE = np.int16  # 16-bit PCM (little-endian assumed by default)
INPUT_BLOCK_DURATION_MS = 50 # Process audio in chunks (milliseconds)
INPUT_BLOCK_SIZE = int(INPUT_SAMPLE_RATE * INPUT_BLOCK_DURATION_MS / 1000)

# Audio settings (Output - Speaker)
OUTPUT_SAMPLE_RATE = 24000 # Required by Live API
OUTPUT_CHANNELS = 1
OUTPUT_DTYPE = np.int16 # 16-bit PCM (little-endian assumed by default)

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
# Use the specific model name required by the user and docs
MODEL_NAME = "gemini-2.0-flash-exp"
API_VERSION = "v1alpha" # Specify the API version

# UI Update Interval
UI_UPDATE_INTERVAL = 0.1 # seconds

# Logging Setup (optional, logs to a file)
logging.basicConfig(filename='gemini_live_audio.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

# --- Global State and Communication ---
# Use asyncio-safe queues for communication with the async Gemini task
input_audio_queue_async = asyncio.Queue()
output_audio_queue = queue.Queue() # Output thread remains sync, so standard queue is fine

stop_event = threading.Event()
app_status = "INITIALIZING"
status_lock = threading.Lock()
last_error = None

# --- Audio Input ---
def audio_input_callback(indata, frames, time, status):
    """Callback function for sounddevice input stream."""
    global app_status, last_error # Use global for status updates

    if status:
        logging.warning(f"Input audio status: {status}")
        with status_lock:
            app_status = "ERROR"
            last_error = f"Audio Input Error: {status}"
            stop_event.set() # Critical error, stop everything
        return # Don't queue data if there's an error

    if not stop_event.is_set():
        # Put data into the asyncio queue from this sync thread
        # Use run_coroutine_threadsafe for thread safety with asyncio queue
        try:
            # Get the loop running in the gemini thread (if it exists)
            loop = getattr(threading.current_thread(), '_loop', None)
            if loop and loop.is_running():
                 asyncio.run_coroutine_threadsafe(input_audio_queue_async.put(indata.tobytes()), loop)
            else:
                # Fallback or handle case where loop isn't available yet/anymore
                # This might happen during startup/shutdown, logging is useful
                logging.warning("Asyncio loop not found or not running for input queue.")
                # A simple non-blocking put might be okay if the consumer is fast
                # but run_coroutine_threadsafe is preferred
                # try:
                #    input_audio_queue_async.put_nowait(indata.tobytes())
                # except asyncio.QueueFull:
                #    logging.warning("Async input queue full, dropping audio frame.")

        except Exception as e:
             logging.error(f"Error putting data into async queue: {e}")


def input_audio_thread_func():
    """Thread function to manage the audio input stream."""
    global app_status, last_error # Use global for status updates
    # Store the loop for the callback
    threading.current_thread()._loop = getattr(threading.main_thread(), '_gemini_loop', None)

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
                 # Only set to LISTENING if not already in error
                 if app_status != "ERROR":
                     app_status = "LISTENING"
            stop_event.wait() # Keep stream open until stop_event is set
    except sd.PortAudioError as pae:
         logging.exception("PortAudio error in input audio thread (often device issue):")
         with status_lock:
            app_status = "ERROR"
            last_error = f"Input Device Error: {pae}"
         stop_event.set()
    except Exception as e:
        logging.exception("Error in input audio thread:")
        with status_lock:
            app_status = "ERROR"
            last_error = f"Input Thread Failed: {e}"
        stop_event.set() # Signal other threads to stop on critical error
    finally:
        logging.info("Input audio thread finished.")

# --- Audio Output ---
def output_audio_thread_func():
    """Thread function to manage the audio output stream."""
    global app_status, last_error # Use global for status updates
    logging.info("Output audio thread started.")
    stream = None # Initialize stream variable
    try:
        stream = sd.OutputStream(
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=OUTPUT_CHANNELS,
            dtype=OUTPUT_DTYPE
        )
        stream.start() # Start the stream explicitly
        logging.info("Audio output stream started.")

        while not stop_event.is_set():
            try:
                # Get audio data from queue, with timeout
                audio_data_bytes = output_audio_queue.get(timeout=0.1)
                if audio_data_bytes:
                    # If we get data, we are potentially speaking
                    with status_lock:
                        if app_status != "ERROR": # Don't override error status
                           app_status = "SPEAKING"
                    # Convert bytes back to NumPy array for playback
                    audio_data_np = np.frombuffer(audio_data_bytes, dtype=OUTPUT_DTYPE)
                    stream.write(audio_data_np)
                    output_audio_queue.task_done() # Signal task completion
                # No 'else' needed, timeout handles the case of no data

            except queue.Empty:
                # No data received in the timeout period
                with status_lock:
                    # If we were speaking but the queue is empty, transition back to listening
                    # unless there's an error state.
                    if app_status == "SPEAKING":
                        app_status = "LISTENING"
                continue # Continue waiting for data
            except sd.PortAudioError as pae:
                 logging.exception("PortAudio error during audio playback:")
                 with status_lock:
                    app_status = "ERROR"
                    last_error = f"Playback Device Error: {pae}"
                 stop_event.set() # Stop on playback device errors
                 break # Exit loop
            except Exception as e:
                 logging.exception("Error during audio playback:")
                 with status_lock:
                     app_status = "ERROR"
                     last_error = f"Playback Error: {e}"
                 # Optional: stop_event.set() ? Decide if playback errors are fatal
                 # Let's make it fatal for now
                 stop_event.set()
                 break # Exit loop

    except sd.PortAudioError as pae:
         logging.exception("PortAudio error initializing output audio stream:")
         with status_lock:
            app_status = "ERROR"
            last_error = f"Output Device Init Error: {pae}"
         stop_event.set()
    except Exception as e:
        logging.exception("Error initializing output audio stream:")
        with status_lock:
            app_status = "ERROR"
            last_error = f"Output Thread Failed: {e}"
        stop_event.set() # Signal other threads to stop
    finally:
        if stream:
             try:
                 # Abort stream to prevent potential hangs waiting for buffer
                 stream.abort(ignore_errors=True)
                 stream.close(ignore_errors=True)
                 logging.info("Audio output stream stopped and closed.")
             except Exception as e_close:
                  logging.error(f"Error closing output stream: {e_close}")
        # Ensure queue is cleared if stopping abnormally
        while not output_audio_queue.empty():
            try:
                output_audio_queue.get_nowait()
                output_audio_queue.task_done()
            except queue.Empty: break
        logging.info("Output audio thread finished.")


# --- Gemini Interaction (Async) ---
async def send_audio_task(session, loop):
    """Async task to send audio chunks from the async queue to Gemini."""
    logging.info("Send audio task started.")
    while not stop_event.is_set():
        try:
            # Get audio chunk from the asyncio queue
            chunk = await asyncio.wait_for(input_audio_queue_async.get(), timeout=0.1)
            if chunk:
                # logging.debug(f"Sending audio chunk: {len(chunk)} bytes")
                # Send using session.send with raw bytes for realtime input
                await session.send(input=chunk)
                input_audio_queue_async.task_done()
            # Optional small sleep if queue is processed very quickly
            # await asyncio.sleep(0.01)
        except asyncio.TimeoutError:
            # No data in queue, continue checking stop_event
            continue
        except Exception as e:
            logging.exception("Error in send audio task:")
            # Signal error and stop, let the main task handle status
            stop_event.set()
            break
    logging.info("Send audio task finished.")

async def receive_audio_task(session, loop):
    """Async task to receive audio chunks from Gemini and put them in the output queue."""
    global app_status, last_error # Use global for status updates
    logging.info("Receive audio task started.")
    try:
        async for response in session.receive():
            if stop_event.is_set():
                logging.info("Stop event set, breaking receive loop.")
                break

            if response.data:
                # logging.debug(f"Received audio data: {len(response.data)} bytes")
                # Put data into the synchronous output queue using run_in_executor
                await loop.run_in_executor(None, output_audio_queue.put, response.data)
            elif response.server_content:
                 if response.server_content.interrupted:
                      logging.warning("Gemini response interrupted (likely by user speech).")
                      # Potentially clear output queue or take other action?
                      # For now, just log it. The output thread will naturally stop playing.
                      # Ensure status reflects listening if we were speaking
                      with status_lock:
                          if app_status == "SPEAKING":
                              app_status = "LISTENING"
                 if response.server_content.model_turn and response.server_content.model_turn.parts:
                      # Although we requested AUDIO, text parts might come for errors/metadata
                      for part in response.server_content.model_turn.parts:
                          if part.text:
                               logging.info(f"Received text from model: {part.text}")
            # Handle other potential responses if needed (tool calls, errors etc.)
            # Example: Check for explicit errors from the server
            # elif hasattr(response, 'error'):
            #     logging.error(f"Received error from Gemini: {response.error}")
            #     with status_lock:
            #          app_status = "ERROR"
            #          last_error = f"Gemini API Error: {response.error}"
            #     stop_event.set()
            #     break


    except Exception as e:
        # Handle potential errors during receive (e.g., connection closed)
        logging.exception("Error in receive audio task:")
        with status_lock:
            app_status = "ERROR"
            last_error = f"Gemini Receive Error: {e}"
        stop_event.set()
    finally:
        logging.info("Receive audio task finished.")


async def gemini_interaction_task():
    """Async task handling the main Gemini Live API interaction."""
    global app_status, last_error # Use global for status updates
    logging.info("Gemini interaction task started.")

    if not GEMINI_API_KEY:
        logging.error("GOOGLE_API_KEY environment variable not set.")
        with status_lock:
            app_status = "ERROR"
            last_error = "API Key Not Found"
        stop_event.set()
        return

    try:
        # Configure the client with the v1alpha version
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': API_VERSION})

        # Define the configuration for the live session
        # Use types for structured configuration
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"], # Request audio output
            # Add system instructions if desired
            system_instruction=types.Content(
                parts=[types.Part(text="You are a helpful and concise voice assistant.")]
            ),
            # SpeechConfig can be added here for voice selection if needed
            # speech_config=types.SpeechConfig(...)
        )

        logging.info(f"Connecting to Gemini Live API with model {MODEL_NAME}...")
        # Establish the asynchronous connection
        async with client.aio.live.connect(model=MODEL_NAME, config=config) as session:
            logging.info("Gemini Live API connection established.")
            with status_lock:
                 # Set status to LISTENING once connected, if not in error
                 if app_status != "ERROR":
                     app_status = "LISTENING"

            loop = asyncio.get_running_loop()

            # Run send and receive tasks concurrently
            send_task_handle = asyncio.create_task(send_audio_task(session, loop))
            receive_task_handle = asyncio.create_task(receive_audio_task(session, loop))

            # Wait for either task to complete (or stop_event to be set)
            done, pending = await asyncio.wait(
                [send_task_handle, receive_task_handle, loop.run_in_executor(None, stop_event.wait)],
                return_when=asyncio.FIRST_COMPLETED
            )

            # If stop_event caused completion, signal others
            if stop_event.is_set():
                 logging.info("Stop event detected, cancelling async tasks.")
            else:
                # One of the tasks finished (possibly due to error), signal stop
                logging.info("An async task finished unexpectedly, signaling stop.")
                stop_event.set()

            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task # Allow cancellation to propagate
                except asyncio.CancelledError:
                    pass # Expected

            # Check results of completed tasks for exceptions
            for task in done:
                 if task.exception():
                      exc = task.exception()
                      logging.error(f"Exception in completed async task: {exc}")
                      # Update status if not already set by the task itself
                      with status_lock:
                          if app_status != "ERROR":
                               app_status = "ERROR"
                               last_error = f"Async Task Error: {exc}"

    except types.PermissionDeniedError as pde:
        logging.exception("Permission denied error connecting to Gemini API (check API key/permissions):")
        with status_lock:
            app_status = "ERROR"
            last_error = f"API Permission Denied: {pde}"
        stop_event.set()
    except Exception as e:
        logging.exception("Error in Gemini interaction task:")
        with status_lock:
            # Avoid overwriting a more specific error from sub-tasks if possible
            if app_status != "ERROR":
                 app_status = "ERROR"
                 last_error = f"Gemini Task Error: {e}"
        stop_event.set()
    finally:
        logging.info("Gemini interaction task finished.")
        # Ensure stop is signaled definitively
        stop_event.set()

def gemini_thread_func():
    """Thread function to run the asyncio event loop for Gemini interaction."""
    logging.info("Gemini API thread started.")
    loop = None
    try:
        # Create and set a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Store the loop reference for the input callback
        threading.main_thread()._gemini_loop = loop
        loop.run_until_complete(gemini_interaction_task())
    except Exception as e:
        logging.exception("Exception in gemini_thread_func running the async loop:")
        stop_event.set() # Ensure stop on loop error
    finally:
        if loop:
             try:
                 # Clean up the loop's resources
                 loop.run_until_complete(loop.shutdown_asyncgens())
                 loop.close()
                 logging.info("Asyncio event loop closed.")
             except Exception as e_close:
                 logging.error(f"Error closing asyncio loop: {e_close}")
        logging.info("Gemini API thread finished.")
        # Clear the loop reference
        if hasattr(threading.main_thread(), '_gemini_loop'):
             delattr(threading.main_thread(), '_gemini_loop')


# --- Curses UI ---
def draw_ui(stdscr):
    """Draws the full-screen UI using curses."""
    # Make getch non-blocking
    stdscr.nodelay(True)
    # Refresh interval for getch, also serves as UI update rate limiter
    stdscr.timeout(int(UI_UPDATE_INTERVAL * 1000))
    curses.curs_set(0) # Hide cursor

    # Initialize colors (check if terminal supports color)
    has_color = curses.has_colors()
    if has_color:
        curses.start_color()
        # Use default background (-1) for better compatibility
        curses.init_pair(1, curses.COLOR_RED, -1)    # Error
        curses.init_pair(2, curses.COLOR_GREEN, -1)  # Listening
        curses.init_pair(3, curses.COLOR_YELLOW, -1) # Speaking (changed from blue for visibility)
        curses.init_pair(4, curses.COLOR_CYAN, -1)   # Initializing / Connecting
        default_color = curses.color_pair(0)
        error_color = curses.color_pair(1) | curses.A_BOLD
        listen_color = curses.color_pair(2)
        speak_color = curses.color_pair(3)
        init_color = curses.color_pair(4)
    else:
        # Define fallbacks if no color
        default_color = curses.A_NORMAL
        error_color = curses.A_BOLD | curses.A_REVERSE
        listen_color = curses.A_NORMAL
        speak_color = curses.A_BOLD
        init_color = curses.A_DIM


    k = 0
    while k != ord('q') and not stop_event.is_set():
        try:
            stdscr.erase()
            height, width = stdscr.getmaxyx()

            # Check minimum size
            if height < 8 or width < 40:
                 stdscr.addstr(0, 0, "Terminal too small...")
                 stdscr.refresh()
                 k = stdscr.getch() # Wait for 'q' in small screen mode
                 if k == ord('q'): break
                 continue # Keep checking size


            # --- Get current status safely ---
            with status_lock:
                current_status = app_status
                current_error = last_error

            # --- Title ---
            title = "Gemini Live Audio Assistant (v1alpha)"
            stdscr.attron(curses.A_BOLD)
            title_x = max(0, (width - len(title)) // 2)
            try: stdscr.addstr(0, title_x, title)
            except curses.error: pass # Ignore if cursor out of bounds
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
                status_text = f"Status: ERROR" # Keep it short
            elif current_status == "INITIALIZING":
                 color = init_color

            try: stdscr.addstr(status_line, 1, status_text.ljust(width - 2), color)
            except curses.error: pass

             # Display error message below status if applicable
            error_line = 3
            if current_status == "ERROR" and current_error:
                 error_display = f"Error: {current_error}"
                 try: stdscr.addstr(error_line, 1, error_display[:width-2], error_color)
                 except curses.error: pass


            # --- Input/Output Queue Info (Optional Debug) ---
            debug_line = 5
            # Safely check asyncio queue size (can estimate)
            in_q_size_approx = input_audio_queue_async.qsize() if input_audio_queue_async else 'N/A'
            out_q_size = output_audio_queue.qsize()
            debug_text = f"In Q (approx): {in_q_size_approx:<5} Out Q: {out_q_size:<5}"
            try: stdscr.addstr(debug_line, 1, debug_text[:width-2])
            except curses.error: pass


            # --- Instructions ---
            instruction_line = height - 2
            instruction = "Press 'q' to quit"
            inst_x = max(0, (width - len(instruction)) // 2)
            try: stdscr.addstr(instruction_line, inst_x, instruction)
            except curses.error: pass

            # --- Border ---
            try: stdscr.border(0)
            except curses.error: pass

            stdscr.refresh()

            # --- Check for input ---
            k = stdscr.getch() # Non-blocking read due to nodelay/timeout

        except curses.error as e:
             # Handle screen resize or other curses issues
             logging.warning(f"Curses error: {e}")
             # Curses auto-handles SIGWINCH for resize usually,
             # but manual redraw might be needed if errors persist
             # For simplicity, we just log and continue, hoping it resolves
             time.sleep(0.5) # Brief pause after error
        except Exception as e:
             logging.exception("Error during curses UI update:")
             stop_event.set() # Stop app on unhandled UI errors
             break # Exit UI loop


    # --- Exit ---
    stop_event.set() # Ensure stop is signalled if 'q' was pressed or loop broken

def main_ui_wrapper(stdscr):
    """Wrapper for curses application setup and teardown."""
    # Setup any initial curses settings if needed (like color)
    draw_ui(stdscr)

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Gemini Live Audio App...")
    # Verify API Key early
    if not GEMINI_API_KEY:
         print("ERROR: GOOGLE_API_KEY environment variable not set.", file=sys.stderr)
         sys.exit(1)

    print(f"Input: {INPUT_SAMPLE_RATE}Hz, Output: {OUTPUT_SAMPLE_RATE}Hz")
    print(f"Using Model: {MODEL_NAME} via API Version: {API_VERSION}")
    print(f"Check {logging.getLogger().handlers[0].baseFilename} for detailed logs.")
    print("Initializing audio devices...")

    # Basic audio device check (optional but helpful)
    try:
        print("Available Input Devices:", sd.query_devices(kind='input'))
        print("Available Output Devices:", sd.query_devices(kind='output'))
        # You could add checks here to ensure default devices exist
    except Exception as e:
        print(f"Warning: Could not query audio devices: {e}")


    # --- Create and start threads ---
    threads = []
    try:
        # Gemini thread runs the asyncio loop
        gemini_thread = threading.Thread(target=gemini_thread_func, name="GeminiAsyncThread", daemon=True)
        threads.append(gemini_thread)
        gemini_thread.start()

        # Short delay to allow gemini thread to potentially start its loop
        # so the input thread callback can find it.
        time.sleep(0.2)

        # Input and Output threads remain synchronous
        input_thread = threading.Thread(target=input_audio_thread_func, name="AudioInputThread", daemon=True)
        output_thread = threading.Thread(target=output_audio_thread_func, name="AudioOutputThread", daemon=True)
        threads.extend([input_thread, output_thread])

        input_thread.start()
        output_thread.start()

        # Give threads a moment to initialize before UI starts
        time.sleep(0.5)

        # Start Curses UI in the main thread
        curses.wrapper(main_ui_wrapper)

    except Exception as e:
        logging.exception("Fatal error during app startup or UI execution:")
        print(f"\nApplication failed: {e}", file=sys.stderr)
        stop_event.set()
    finally:
        print("\nShutting down...")
        stop_event.set() # Ensure stop is signalled

        # Wait for threads to finish
        for thread in threads:
            try:
                 # Add a slightly longer timeout for potentially blocking IO threads
                 thread.join(timeout=3.0)
                 if thread.is_alive():
                     print(f"Warning: Thread {thread.name} did not finish cleanly.")
            except Exception as e:
                 print(f"Error joining thread {thread.name}: {e}")

        print("Application finished.")