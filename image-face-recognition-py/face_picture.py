import cv2

# Carregando os cascades com as features de faces e rostos
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

# Função para detectar os rostos e desenhar os retângulos
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

# Carregando a imagem
frame = cv2.imread('images/teste.jpeg')

# Inicializando um outro frame P&B (o desempenho é melhor em imagens P&B por conta de menos detalhes para identificar as features)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

canvas = detect(gray, frame)
cv2.imshow('Face detection', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
