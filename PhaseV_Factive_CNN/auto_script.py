import pyautogui as gui
import time

gui.moveTo(404, 1061, duration=0.25)
gui.click()

gui.moveTo(612, 24, duration=0.25)
gui.click()

init_PAGE = 234
PAGE = init_PAGE # Starting page
# PAGE = 217 # Starting page

while True:
    # Connection failed
    if gui.pixelMatchesColor(1129, 208, (90, 151, 255)):
        break

    # Select All
    gui.moveTo(355, 405, duration=0.25)
    gui.click()

    # RTF Icon
    gui.moveTo(559, 368, duration=0.25)
    gui.mouseDown()
    time.sleep(1)
    gui.mouseUp()

    # Choose Headline format
    gui.moveTo(605, 392, duration=0.25)
    gui.click()

    # Wait for download interface loading
    wait = 0
    while True:
        if wait > 20:
            # RTF Icon
            gui.moveTo(559, 368, duration=0.25)
            gui.mouseDown()
            time.sleep(1)
            gui.mouseUp()
            # Choose Headline format
            gui.moveTo(605, 392, duration=0.25)
            gui.click()
            wait = 0
            continue
        time.sleep(1)
        wait += 1
        if gui.pixelMatchesColor(771, 171, (24, 25, 31)):
            continue
        else:
            break

    # Input file name
    gui.typewrite(str(PAGE)+'.rtf')

    # Press ENTER
    gui.press('enter')

    # # Wait for downloading
    # gui.PAUSE = 10

    # Unselect All
    gui.moveTo(355, 405, duration=0.25)
    gui.click()

    # Move to next page
    if PAGE == init_PAGE:
        gui.moveTo(595, 408, duration=0.25)
        gui.click()
    else:
        gui.moveTo(725, 410, duration=0.25)
        gui.click()

    # Wait for page loading
    time.sleep(7)

    # Update page number
    PAGE += 1