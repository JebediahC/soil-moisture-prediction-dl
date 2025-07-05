## entrance of the program


from src.utils import SimpleLogger

logger_1 = SimpleLogger(name="Logger-1", log_path=None)
logger_2 = SimpleLogger(name="Logger-2", log_path="logs/logger_2.log")

logger_1.log("This is an info message from Logger-1.")

logger_2.log("This is an info message from Logger-2.")

# spend some time
import time
time.sleep(1)

logger_1.log("This is an info message from Logger-1.")

logger_2.log("This is another info message from Logger-2.")