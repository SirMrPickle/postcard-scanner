import subprocess
import threading


sV = 4
frontScript = "python3 front-scanner-v" + str(sV) + ".py"
backScript = "python3 back-scanner-v" + str(sV) + ".py"

cV = 4
combineScript = "python3 combine-v" + str(cV) + ".py"


def run(script):
    subprocess.run(script, shell=True)


# Phase 1
thread1 = threading.Thread(target=run, args=(frontScript,))
thread2 = threading.Thread(target=run, args=(backScript,))

print(f"Running Scripts: '{frontScript}' and '{backScript}'")
thread1.start()
thread2.start()

# waiting game

thread1.join()
thread2.join()
print(f"Scripts '{frontScript}' and '{backScript}' complete.")

# Phase 2
print(f"Running Script: '{combineScript}'")
subprocess.run(combineScript, shell=True)
print(f"Script `{combineScript}' complete.")
