import face_recognition
import numpy as np 
import cv2
imtr=face_recognition.load_image_file("elon.jpg")
imtr=cv2.cvtColor(imtr,cv2.COLOR_BGR2RGB)
im_resize1=cv2.resize(imtr,(500,400))

imtest=face_recognition.load_image_file("elon_test.jpg")
imtest=cv2.cvtColor(imtest,cv2.COLOR_BGR2RGB)
imt=cv2.resize(imtr,(500,400))



loc=face_recognition.face_locations(im_resize1)[0]
encode=face_recognition.face_encodings(im_resize1)[0]

f1=(loc[3],loc[0])
f2=(loc[1],loc[2])
im_resize1=cv2.rectangle(im_resize1,f1,f2,(255,0,255),2)

loctest=face_recognition.face_locations(imtest)[0]
encodetest=face_recognition.face_encodings(imtest)[0]

f3=(loctest[3],loctest[0])
f4=(loctest[1],loctest[2])
imtest=cv2.rectangle(imtest,f3,f4,(255,0,255),2)


result= face_recognition.compare_faces([encode],encodetest)
print(result)
facedis=face_recognition.face_distance([encode],encodetest)
print(facedis)

cv2.imshow("elon",im_resize1)
cv2.imshow("elon musk",imtest)
cv2.waitKey(0)