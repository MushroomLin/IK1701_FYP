import pyautogui as gui
import time

TOTLE_NUM_OF_FILE = 264
# Minimize Powershell
gui.moveTo((1767,19), duration=0.25)
gui.click()

# # First 32 files
# for i in range(3):
#     # Open rtf file
#     gui.moveTo((1402,164), duration=0.25)
#     gui.click()
#     gui.press('enter')

#     # Wait for rtf file loading
#     while True:
#         time.sleep(0.5)
#         if gui.pixelMatchesColor(1175, 45, (42, 87, 154)):
#             break

#     # Copy
#     gui.keyDown('ctrlleft')
#     gui.press('a')
#     gui.keyUp('ctrlleft')
#     gui.keyDown('ctrlleft')
#     gui.press('c')
#     gui.keyUp('ctrlleft')

#     # Minimize rtf file
#     gui.moveTo((1780,21), duration=0.25)
#     gui.click()

#     # Open txt file
#     gui.moveTo((485,157), duration=0.25)
#     gui.click()
#     gui.press('enter')

    # # click
    # gui.moveTo((415, 270), duration=0.25)
    # gui.click()

#     # Paste, save, exit
#     gui.keyDown('ctrlleft')
#     gui.press('v')
#     gui.keyUp('ctrlleft')
#     gui.keyDown('ctrlleft')
#     gui.press('s')
#     gui.keyUp('ctrlleft')
#     time.sleep(0.5)
#     if gui.pixelMatchesColor(750, 527, (252, 225, 0)) or gui.pixelMatchesColor(750, 518, (252, 225, 0)):
#         gui.moveTo((1062,645))
#         gui.click()
#     gui.keyDown('altleft')
#     gui.press('f4')
#     gui.keyUp('altleft')

#     # Roll down txt dir
#     gui.moveTo((947,970), duration=0.25)
#     gui.click()
#     time.sleep(0.2)
#     gui.click()

#     # Roll down rtf dir
#     gui.moveTo((1907,970), duration=0.25)
#     gui.click()
#     time.sleep(0.2)
#     gui.click()

#     # Close rtf file
#     gui.moveTo((1022,1063), duration=0.25)
#     gui.click()
#     gui.keyDown('altleft')
#     gui.press('f4')
#     gui.keyUp('altleft')

# # The rest files
# gui.moveTo((946,956), duration=0.25)
# gui.mouseDown()
# time.sleep(5)
# gui.mouseUp()
# gui.moveTo((1908,956), duration=0.25)
# gui.mouseDown()
# time.sleep(5)
# gui.mouseUp()

for i in range(TOTLE_NUM_OF_FILE - (TOTLE_NUM_OF_FILE-24)):
    # Open rtf file
    gui.moveTo((1466,970), duration=0.25)
    gui.click()
    gui.press('enter')

    # Wait for rtf file loading
    while True:
        time.sleep(0.5)
        if gui.pixelMatchesColor(1175, 45, (42, 87, 154)):
            break

    # Copy
    gui.keyDown('ctrlleft')
    gui.press('a')
    gui.keyUp('ctrlleft')
    gui.keyDown('ctrlleft')
    gui.press('c')
    gui.keyUp('ctrlleft')

    # Minimize rtf file
    gui.moveTo((1780,21), duration=0.25)
    gui.click()

    # Open txt file
    gui.moveTo((500,970), duration=0.25)
    gui.click()
    gui.press('enter')

    # click
    gui.moveTo((415, 270), duration=0.25)
    gui.click()

    # Paste, save, exit
    gui.keyDown('ctrlleft')
    gui.press('v')
    gui.keyUp('ctrlleft')
    gui.keyDown('ctrlleft')
    gui.press('s')
    gui.keyUp('ctrlleft')
    time.sleep(0.5)
    if gui.pixelMatchesColor(750, 518, (252, 225, 0)) or gui.pixelMatchesColor(750, 527, (252, 225, 0)):
        gui.moveTo((1062,645))
        gui.click()
    gui.keyDown('altleft')
    gui.press('f4')
    gui.keyUp('altleft')

    # Roll up txt dir
    gui.moveTo((947,127), duration=0.25)
    gui.click()
    time.sleep(0.2)
    gui.click()

    # Roll down rtf dir
    gui.moveTo((1907,127), duration=0.25)
    gui.click()
    time.sleep(0.2)
    gui.click()

    # Close rtf file
    gui.moveTo((1022,1063), duration=0.25)
    gui.click()
    gui.keyDown('altleft')
    gui.press('f4')
    gui.keyUp('altleft')