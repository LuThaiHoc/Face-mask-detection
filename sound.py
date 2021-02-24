import time
from multiprocessing import Process
from playsound import playsound


sound_file = 'sound/warning2.mp3'

def playy():
    playsound(sound_file)


t = time.time()

stop_thread = False

def play_sound():  
    global t
    if time.time() - t > 5:
        P = Process(name="playsound",target=playy)
        P.start() # Inititialize Process
        


# def play():
#     playsound(wavFile)


# def play_sound():
#     sound_thread = threading.Thread(target=play)
#     sound_thread.start()
#     sound_thread.join()

