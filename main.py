import cv2
import mediapipe as mp 
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap=cv2.VideoCapture(0) 
up=False
count=0
while True:
    success,img = cap.read()
    img = cv2.resize(img,(1280,720))
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    result=pose.process(img_RGB)
    if result.pose_landmarks:
        mpDraw.draw_landmarks(img,result.pose_landmarks,mpPose.POSE_CONNECTIONS)
        points={}
        for id,lm in enumerate(result.pose_landmarks.landmark):
            h,w,c=img.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            points[id]=(cx,cy)
        
        cv2.circle(img,points[11],15,(255,0,0),cv2.FILLED)
        cv2.circle(img,points[12],15,(255,0,0),cv2.FILLED)
        cv2.circle(img,points[13],15,(255,0,0),cv2.FILLED)
        cv2.circle(img,points[14],15,(255,0,0),cv2.FILLED)
        if not up and points[14][1] < points[12][1]:
            print("up")
            up=True
            count+=1
        elif points[14][1]>points[12][1]:
            print("Down")
            up=False
    cv2.putText(img,str(count),(100,150),cv2.FONT_HERSHEY_PLAIN,12,(255,0,0),12)
    
            
    cv2.imshow("image",img)         
    key=cv2.waitKey(1)&0xFF
    if key==27:
        break
if count==0:
    print("We all start somewhere, don't give up and keep pushing yourself")
else:
    print("You did",count,"repetitions,Well Done!")
cap.release()
cv2.destroyAllWindows()
