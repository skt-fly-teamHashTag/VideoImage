import numpy as np 
import cv2 
'''
img = np.zeros((200, 300), np.uint8)

img[:] = 200 

title1, title2 = 'poisition1', 'position2'
cv2.namedWindow(title1, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(title2)
cv2.moveWindow(title1, 150, 150)
cv2.moveWindow(title2, 200, 300)

cv2.imshow(title1, img)
cv2.imshow(title2, img)

cv2.resizeWindow(title1, 400, 300)
cv2.resizeWindow(title2, 400, 300)
cv2.waitKey(0)
cv2.destroyAllWindows()

switch_case = {
    ord('a'): "a키 입력",
    ord('b'): "b키 입력", 
    0x41: "A키 입력",
    int('0x42', 16): "B키 입력", 
    2424832: "왼쪽 화살표키 입력", 
    2490368: "윗쪽 화살표키 입력", 
    2555904: "오른쪽 화살표키 입력", 
    2621440: "아래쪽 화살표키 입력"
} 

img = np.ones((200, 300), np.float64)
cv2.namedWindow('keyboard Event')
cv2.imshow('keyboard Event', img)

while True: 
    key = cv2.waitKeyEx(100)
    if key == 27: break 

    try:
        result = switch_case[key]
        print(result)
    except KeyError:
        result = -1
cv2.destroyAllWindows()

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("마우스 왼쪽 버튼 누르기")
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("마우스 오른족 버튼 누르기")
    elif event == cv2.EVENT_RBUTTONUP:
        print("마우스 오른쪽 버튼 떼기")
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        print("마우스 왼쪽 버튼 더블클릭")

img = np.full((200, 300), 255, np.uint8)


title1, title2 = 'mouse event1', 'mouse event2'
cv2.imshow(title1, img)
cv2.imshow(title2, img)

cv2.setMouseCallback('mouse event1', onMouse)

cv2.waitKey(0)
cv2.destroyAllWindows()

def onChange(value):
    global image, title 

    add_value = value -int(image[0][0])

def onchange(value):
    global image, titile 

    add_value = value - int(image[0][0])
    image[:] = image + add_value 
    cv2.imshow(titile, image)

def onMouse(event, x, y, flags, param):
    global image, bar_name 

    if event == cv2.EVENT_RBUTTONDOWN:
        print(image[0][0])
        if image[0][0] 


###230131
blue, greeen, red = (255, 0, 0), (0, 255, 0), (0, 0, 255)
image = np.zeros((400, 600, 3), np.uint8)
image[:] = (255, 255, 255)

pt1, pt2 = (50, 50), (250, 350)
pt3, pt4 = (400, 150), (500, 50)
roi = (50, 200, 200, 100)

# 직선 그리기 
orange, blue,cyan = (0, 165, 255), (255, 0, 0), (255, 255, 0)
white, black = (255, 255, 255), (0, 0, 0)
image = np.full((300, 500, 3), white, np.uint8)

center = (image.shape[1]//2, image.shape[0]//2)
pt1, pt2 = (300, 50), (100, 200)
size = (120, 60)

cv2.circle(image, pt1, 1, 0, 2)
cv2.circle(image, pt2, 1, 0, 2)

cv2.ellipse(image, pt1, size, 0, 0, 360, blue, 1)
cv2.ellipse(image, pt2, size, 90, 0, 360, blue, 1)

## 마우스 이벤트 및 그리기 
def onMouse(event, x, y, flags, param):
    global title, pt

    if event == cv2.EVENT_LBUTTONDOWN:
        if pt[0] < 0: pt = (x, y)
        else:
            cv2.rectangle(image, pt, (x, y), (255, 0, 0), 2)
            cv2.imshow(title, image)
            pt = (-1, -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if pt[0] < 0: pt = (x, y)
        else: 
            dx, dy = pt[0]-x, pt[1]-y
            radius = int(np.sqrt(dx*dx+dy*dy))
            cv2.circle(image, pt, radius, (0, 0, 255), 2)
            cv2.imshow(title, image)
            pt = (-1, -1)

image = np.full((300, 500, 3), (255, 255, 255), np.uint8)

pt = (-1, -1) 
title = "Draw Event" 
cv2.imshow(title, image)
cv2.setMouseCallback(title, onMouse)
cv2.waitKey(0)


image = np.zeros((300, 400, 3), np.uint8)
image[:] = (255, 255, 255)

pt1, pt2 = (50, 130), (200, 300)

cv2.line(image, pt1, (100, 200))


## 예제 
def print_matInfo(name, image):
    if image.dtype == 'uint8':     mat_type = "CV_8U"
    elif image.dtype == 'int8':    mat_type = "CV_8S"
    elif image.dtype == 'uint16':  mat_type = "CV_16U"
    elif image.dtype == 'int16':   mat_type = "CV_16S"
    elif image.dtype == 'float32': mat_type = "CV_32F"
    elif image.dtype == 'float64': mat_type = "CV_64F"
    nchannel = 3 if image.ndim == 3 else 1

    ## depth, channel 출력
    print("%12s: depth(%s), channels(%s) -> mat_type(%sC%d)"
          % (name, image.dtype, nchannel, mat_type,  nchannel))

title1, title2 = 'gray2gray', 'gray2color'
gray2gray = cv2.imread("images/read_gray.jpg", cv2.IMREAD_GRAYSCALE)



image = cv2.imread("iimages/read_color.jpg", cv2.IMREAD_COLOR)
if image is None: raise Exception("영상 파일 읽기 에러")

params_jpg = (cv)
'''
## 비디오 
def put_string(frame, text, pt, value, color = (120, 200, 90)):
    text += str(value)
    shade = (pt[0]+2, pt[1]+2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, shade, font, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, text, pt, font, 0.7, color, 2)

capture = cv2.VideoCapture(0)
if capture.isOpened() == False:
    raise Exception("카메라 연결 안됨")

def zoom_bar(value):
    global capture 
    capture.set(cv2.CAP_PROP_ZOOM, value)

def focus_bar(value):
    global capture 
    capture.set(cv2.CAP_PROP_FOCUS, value)

capture = cv2.VideoCapture(0)
if capture.isOpened() == False: raise Exception("카메라 연결 안됨")

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
capture.set(cv2.CAP_PROP_BRIGHTNESS, 100)

title = 'Change camera propreties'
cv2.namedWindow(title)
cv2.createTrackbar('zoom', title, 0, 10, zoom_bar)
cv2.createTrackbar('focus', title, 0, 40, focus_bar)

while True: 
    ret, frame = capture.read()
    if not ret: break 
    if cv2.waitKey(30) >=0: break 

    zoom = cv2.getTrackbarPos("zoom", title)
    focus = cv2.getTrackbarPos("focus", title)
    put_string(frame, 'zoom: ', (10, 240), zoom)
    put_string(frame, 'focus: ', (10, 270), focus)
    cv2.imshow(title, frame)
capture.release()

## 영상 저장 
fps = 29.97 
delay = round(1000/fps)
size = (640, 360)
fourcc = cv2.VideoWriter_fourcc(*'DX50')

print("width * height: ", size)
print("VideoWriterFourcc: %s"% fourcc)
print("delay: %2d ms "%delay)
print("fps: %.2f"%fps)

capture.set(cv2.CAP_PROP_ZOOM, 1)
capture.set(cv2.CAP_PROP_FOCUS, 0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

writer = cv2.VideoWriter("images/video_file.avi", fourcc, fps, size)
if writer.isOpened() == False: raise Exception("동영상 파일 개방 안됨")

while True:
    ret, frame = capture.read()
    if not ret: break 
    if cv2.waitKey(delay) >= 0: break 

    writer.write(frame)
    cv2.imshow("vies frame from camera", frame)

writer.release()
capture.relese()

## 
image = cv2.imread("images/flip_test.jpg", cv2.IMREAD_COLOR)
if image is None: raise Exception("영상파일 읽어보기 오류 발생")

x_axis = cv2.flip(image, 0)
y_axis = cv2.flip(image, 1)
xy_axis = cv2.flip(image, -1)
rep_image = cv2.repeat(image, 1, 2)
trans_image = cv2.transpose(image)

titles = ['image', 'x_axis', 'y_axis', 'xy_axis', 'rep_image', 'trans_image']
for title in titles:
    cv2.imshow(title, eval(title))

cv2.waitKey(0)


## 
ch0 = np.zeros((2, 4), np.uint8)+10
ch1 = np.ones((2, 4), np.uint8)*20
ch2 = np.full((2, 4), 30, np.uint8)

list_bgr = [ch0, ch1, ch2]
merge_bgr = cv2.merge(list_bgr)
split_bgr = cv2.split(merge_bgr)

print("split_bgr 행렬형태", np.array(split_bgr).shape)
print("merge_bgr 행렬형태", merge_bgr.shape)
print("[ch0]= \n%s" %ch0)
print("[ch1]= \n%s" %ch1)
print("[ch2]= \n%s" %ch2)
print()


##230201 
# 지수, 로그, 제곱근 관련 함수 
x = np.array([1, 2, 3, 5, 10], np.float32)
y = np.array([2, 5, 7, 2, 9]).astype("float32")

mag = cv2.magnitude(x, y)
ang = cv2.phase(x, y)
p_mag, p_ang = cv2.cartToPolar(x, y)
x2, y2 = cv2.polarToCart(p_mag, p_ang)

print("[x] 형태: %s 원소: %s"(x.shape, x))
print("[mag 형태]")


image1 = np.zeros((300, 300), np.uint8)           # 300행, 300열 검은색 영상 생성
image2 = image1.copy()

h, w = image1.shape[:2]
cx,cy  = w//2, h//2
cv2.circle(image1, (cx,cy), 100, 255, -1)            # 중심에 원 그리기
cv2.rectangle(image2, (0,0, cx, h), 255, -1)

image3 = cv2.bitwise_or(image1, image2)        # 원소 간 논리합
image4 = cv2.bitwise_and(image1, image2)       # 원소 간 논리곱
image5 = cv2.bitwise_xor(image1, image2)       # 원소 간 배타적 논리합
image6 = cv2.bitwise_not(image1)               # 행렬 반전

cv2.imshow("image1", image1);         cv2.imshow("image2", image2)
cv2.imshow("bitwise_or", image3);      cv2.imshow("bitwise_and", image4)
cv2.imshow("bitwise_xor", image5);   cv2.imshow("bitwise_not", image6)
cv2.waitKey(0)

# 예제 
image = cv2.imread("images/color/jpg", cv2.IMREAD_COLOR)

pt1, pt2 = (0, 0), (15, 20)
pt3, pt4 = (15, 20), (40, 40)
