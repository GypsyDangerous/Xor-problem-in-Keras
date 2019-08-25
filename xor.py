from misc import *
from keras.models import Model 
from keras.layers import Input, Dense, Dropout
import matplotlib.pyplot as plt
import matplotlib.animation as animation

x_train = np.array([
			[1, 0],
			[0, 1],
			[1, 1],
			[0, 0]])

y_train = np.array([[1], [1], [0], [0]])


input_img = Input(shape=x_train[0].shape)

hidden = Dense(8, activation='relu')(input_img)
hidden = Dropout(.25)(hidden)

hidden = Dense(4, activation='relu')(hidden)

output = Dense(1, activation='sigmoid')(hidden)

xor = Model(input_img, output)

xor.compile(optimizer="adadelta", loss="binary_crossentropy", metrics=['accuracy'])

# xor.fit(x_train, y_train,
# 	epochs=1000,
# 	shuffle=True,
# 	batch_size=256)
guess = xor.predict(x_train)
print(guess)

fig = plt.figure()
ims = []
size = 50
for i in range(200):
	print(f"Frame: {i}")
	xor.fit(x_train, y_train,
	epochs=10,
	shuffle=True,
	batch_size=256)

	img = np.zeros((size, size))
	for i in range(size):
		for j in range(size):
			data = [[[i/size, j/size]]]
			img[i, j] = xor.predict(data)[0]/2
	im = plt.imshow(img, cmap="gray", animated=True)
	ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

ani.save(f'C:\\Users\\snyde\\Videos\\xor{size}.mp4')
plt.show()
