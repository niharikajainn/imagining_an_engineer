import os
import shlex
import shutil
import crop_resize_faces
import original
import synthetic
import running_total
import time
import appscript

#Clear screen and open Photo Booth app, wait to take picture
os.system('clear')
os.system('osascript -e \"tell application \\\"Photo Booth\\\" to activate\"')
time.sleep(10)

#Close Photo Booth app (to ensure photos are deleted from display)
os.system('osascript -e \'quit app \"Photo Booth\"\'')

#Move photo into directory and pre-process
path1 = "Pictures/Photo Booth Library/Pictures/"
path2 = "Documents/College/Senior Year/Fall/FURI/presentation/photo.jpg"
usr = "../../../../../../"
files = os.listdir(usr + path1)
shutil.move(usr+path1+files[0], usr + path2)
crop_resize_faces.main()

#Run on original classifier and update totals
print("Original data:")
result = original.main()
if result:
	running_total.og_f_eng()
else:
	running_total.og_f_noneng()

#Run on augmented classifier and update totals
print("Synthetic data:")
result = synthetic.main()
if result:
	running_total.aug_f_eng()
else:
	running_total.aug_f_noneng()

#Display totals in new window
#appscript.app('Terminal').do_script('cd Documents/College/Senior\ Year/Fall/FURI/presentation/ && python running_total.py')


