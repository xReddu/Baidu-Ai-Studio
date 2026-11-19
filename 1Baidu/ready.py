import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("digits.jpg").convert("L")   # "L" = оттенки серого
a = np.array(img)
print("Размер изображения:", a.shape)

plt.figure(figsize=(5, 5))
plt.imshow(a, cmap="gray")
plt.title("Исходное изображение: 10 строк, цифры 0–9")
plt.axis("off")
plt.show()




ink = a < 110                      # чёрные штрихи = True

def bands(p, gap):                 # непрерывные участки проекции
    idx = np.where(p > 0)[0]; out = []; s = prev = idx[0]
    for i in idx[1:]:
        if i - prev > gap: out.append((s, prev + 1)); s = i
        prev = i
    return out + [(s, prev + 1)]

rows = bands(ink.sum(1), 20)       # строки: 10 штук = цифры 0..9
print("Найдено строк:", len(rows))




X, y, originals = [], [], []
for label, (r0, r1) in enumerate(rows):
    for c0, c1 in bands(ink[r0:r1].sum(0), 14):
        if c1 - c0 < 8 or c1 > 1195: continue          # мусор и обрезанный край
        cell = Image.fromarray((ink[r0:r1, c0:c1] * 255).astype("uint8"))
        originals.append(cell.copy())                  # для показа "Оригинал"
        cell.thumbnail((20, 20))
        canvas = Image.new("L", (28, 28))
        canvas.paste(cell, ((28 - cell.width) // 2, (28 - cell.height) // 2))
        X.append(np.array(canvas).ravel() / 255)
        y.append(label)
X, y = np.array(X), np.array(y)
print("Форма данных X:", X.shape, "| меток y:", y.shape)
print("Образцов на каждую цифру:", np.bincount(y))

k = 23                                                  # номер образца, можно менять
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1); plt.imshow(originals[k], cmap="gray"); plt.title("Оригинал"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(X[k].reshape(28, 28), cmap="gray")
plt.title("После преобразования\nв 28×28 и нормализации"); plt.axis("off")
plt.show()
print("Метка (правильная цифра):", y[k])
print("Среднее значение после нормализации:", X[k].mean())





import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from sklearn.model_selection import train_test_split

paddle.seed(0)                                   # чтобы результат повторялся
print("Версия PaddlePaddle:", paddle.__version__)
print("Устройство:", paddle.device.get_device())

# те же 70% / 30%, что и раньше
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
xt = paddle.to_tensor(Xtr, dtype="float32")
yt = paddle.to_tensor(ytr.reshape(-1, 1), dtype="int64")
xe = paddle.to_tensor(Xte, dtype="float32")
print("Тензор обучения:", xt.shape, "| тензор теста:", xe.shape)

# маленькая нейросеть: 784 пикселя → 64 нейрона → 10 цифр
model = nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))
opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())

for epoch in range(1, 201):
    loss = F.cross_entropy(model(xt), yt)
    loss.backward()
    opt.step()
    opt.clear_grad()
    if epoch % 40 == 0:
        print("Эпоха {}: ошибка (loss) = {:.4f}".format(epoch, float(loss)))

model.eval()
pred_te = model(xe).argmax(axis=1).numpy()
print("Точность на тесте: {:.0%}".format((pred_te == yte).mean()))




i = 5                                                   # номер тестового образца, можно менять
proba = model.predict_proba(Xte[i:i+1])[0]
pred = int(proba.argmax())

plt.figure(figsize=(3, 3))
plt.imshow(Xte[i].reshape(28, 28), cmap="gray")
plt.title("Это цифра: {}".format(pred), fontsize=16)
plt.axis("off"); plt.show()

print("Модель распознала цифру:", pred)
print("Правильный ответ:", yte[i])
print("Уверенность: {:.0%}".format(proba[pred]))



pred_all = model.predict(Xte)
plt.figure(figsize=(12, 4))
for j in range(20):
    plt.subplot(2, 10, j + 1)
    plt.imshow(Xte[j].reshape(28, 28), cmap="gray")
    plt.title(str(pred_all[j]), color="green" if pred_all[j] == yte[j] else "red")
    plt.axis("off")
plt.suptitle("Что распознала модель (зелёный — верно, красный — ошибка)")
plt.show()