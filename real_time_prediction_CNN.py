
import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo
model = tf.keras.models.load_model("my_model.h5")

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocesar la imagen (mantener en RGB)
        resized = cv2.resize(frame, (28, 28))
        normalized = resized / 255.0
        input_img = normalized.reshape(1, 28, 28, 3)

        # Predecir
        prediction = model.predict(input_img, verbose=0)
        predicted_label = np.argmax(prediction)
        prediction_probs = prediction.flatten()

        # Clonar frame para dibujo
        frame_with_prediction = frame.copy()

        # Mostrar todas las probabilidades
        for i, prob in enumerate(prediction_probs):
            text = f"{i}: {prob:.2f}"
            color = (0, 255, 0) if i == predicted_label else (0, 0, 255)
            y_position = 30 + i * 25
            cv2.putText(frame_with_prediction, text, (10, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Mostrar el número predicho de forma grande
        cv2.putText(frame_with_prediction, f"Pred: {predicted_label}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)

        # Mostrar ventanas
        cv2.imshow('Real-time prediction', frame_with_prediction)

        # Mostrar la imagen reescalada de entrada
        upscaled_resized = cv2.resize(normalized, (280, 280), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Resized Input (RGB)', upscaled_resized)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
