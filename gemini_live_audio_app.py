def draw_ui(stdscr):
    """Draws the full-screen UI using curses."""
    # Make getch non-blocking
    stdscr.nodelay(True)
    # Refresh interval for getch, also serves as UI update rate limiter
    stdscr.timeout(int(UI_UPDATE_INTERVAL * 1000))
    curses.curs_set(0) # Hide cursor

    # --- Color Setup ---
    use_colors = False # Default to no colors
    default_color, error_color, listen_color, speak_color, init_color = (curses.A_NORMAL,) * 5

    try:
        if curses.has_colors():
            curses.start_color()
            # Check if start_color worked (optional, but can help debug)
            if curses.can_change_color(): # Or another check if start_color was effective
                 # Use default background (-1) for better compatibility
                curses.init_pair(1, curses.COLOR_RED, -1)    # Error
                curses.init_pair(2, curses.COLOR_GREEN, -1)  # Listening
                curses.init_pair(3, curses.COLOR_YELLOW, -1) # Speaking
                curses.init_pair(4, curses.COLOR_CYAN, -1)   # Initializing / Connecting

                # Assign colors if init succeeded
                default_color = curses.color_pair(0)
                error_color = curses.color_pair(1) | curses.A_BOLD
                listen_color = curses.color_pair(2)
                speak_color = curses.color_pair(3)
                init_color = curses.color_pair(4)
                use_colors = True # Colors are successfully initialized
            else:
                 logging.warning("Terminal reports has_colors but colors might not be fully usable.")
                 # Fallback attributes already set outside the try block

    except curses.error as e:
        logging.warning(f"Curses color initialization failed: {e}. Falling back to non-color mode.")
        # Fallback attributes are already set outside the try block

    # --- Fallback definitions if colors failed or not supported ---
    if not use_colors:
        default_color = curses.A_NORMAL
        error_color = curses.A_BOLD | curses.A_REVERSE # Make error standout without color
        listen_color = curses.A_NORMAL
        speak_color = curses.A_BOLD
        init_color = curses.A_DIM # Dim for initializing state


    k = 0
    # --- Main UI Loop (rest of the function remains largely the same) ---
    while k != ord('q') and not stop_event.is_set():
        try:
            stdscr.erase()
            height, width = stdscr.getmaxyx()

            # Check minimum size
            if height < 8 or width < 40:
                 # Use default_color for the small screen message
                 try: stdscr.addstr(0, 0, "Terminal too small...", default_color)
                 except curses.error: pass # Ignore errors if even this fails
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
             # Use default_color for title base, apply bold separately
            title_x = max(0, (width - len(title)) // 2)
            try:
                stdscr.attron(curses.A_BOLD)
                stdscr.addstr(0, title_x, title, default_color)
                stdscr.attroff(curses.A_BOLD)
            except curses.error: pass # Ignore if cursor out of bounds


            # --- Status Display ---
            status_line = 2
            status_text = f"Status: {current_status}"
            # Select the appropriate color attribute based on current_status
            color_attr = default_color
            if current_status == "LISTENING":
                color_attr = listen_color
            elif current_status == "SPEAKING":
                color_attr = speak_color
            elif current_status == "ERROR":
                color_attr = error_color
                status_text = f"Status: ERROR" # Keep it short
            elif current_status == "INITIALIZING":
                 color_attr = init_color

            # Ensure status text is padded and drawn with selected attribute
            try: stdscr.addstr(status_line, 1, status_text.ljust(width - 2), color_attr)
            except curses.error: pass

             # Display error message below status if applicable
            error_line = 3
            if current_status == "ERROR" and current_error:
                 error_display = f"Error: {current_error}"
                 # Use the error_color attribute for the error message
                 try: stdscr.addstr(error_line, 1, error_display[:width-2], error_color)
                 except curses.error: pass


            # --- Input/Output Queue Info (Optional Debug) ---
            debug_line = 5
            in_q_size_approx = input_audio_queue_async.qsize() if input_audio_queue_async else 'N/A'
            out_q_size = output_audio_queue.qsize()
            debug_text = f"In Q (approx): {in_q_size_approx:<5} Out Q: {out_q_size:<5}"
             # Use default_color for debug text
            try: stdscr.addstr(debug_line, 1, debug_text[:width-2], default_color)
            except curses.error: pass


            # --- Instructions ---
            instruction_line = height - 2
            instruction = "Press 'q' to quit"
            inst_x = max(0, (width - len(instruction)) // 2)
             # Use default_color for instructions
            try: stdscr.addstr(instruction_line, inst_x, instruction, default_color)
            except curses.error: pass

            # --- Border ---
            # Use default_color for the border
            try: stdscr.border(0) # Border uses default attributes unless specified
            except curses.error: pass

            stdscr.refresh()

            # --- Check for input ---
            k = stdscr.getch() # Non-blocking read due to nodelay/timeout

        except curses.error as e:
             logging.warning(f"Curses error during draw loop: {e}")
             time.sleep(0.5) # Brief pause after error
        except Exception as e:
             logging.exception("Error during curses UI update:")
             stop_event.set() # Stop app on unhandled UI errors
             break # Exit UI loop


    # --- Exit ---
    stop_event.set() # Ensure stop is signalled if 'q' was pressed or loop broken