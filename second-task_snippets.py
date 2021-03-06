model2 = Sequential()
model2.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Hidden layer with 500 nodes
model2.add(Flatten())
model2.add(Dense(500, activation="relu"))
model2.add(Dense(32, activation="softmax"))
model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
shift = 0.2
datagen2 = ImageDataGenerator(rotation_range=15, width_shift_range=shift, height_shift_range=shift)
datagen2.fit(X_train)
history2 = model2.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)
history_flow2 = model2.fit_generator(datagen2.flow(X_train, Y_train, batch_size=32), steps_per_epoch=len(X_train) / 32, epochs=10, verbose=1, validation_data=(X_test, Y_test), validation_steps=len(X_test) / 32)

plt.figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
#  "Accuracy"
plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.plot(history_flow2.history['acc'])
plt.plot(history_flow2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation', 'train_flow', 'val_flow'], loc='lower right')
plt.show()

plt.figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
# "Loss"
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.plot(history_flow2.history['loss'])
plt.plot(history_flow2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation', 'train_flow', 'val_flow'], loc='upper right')
plt.show()

# network trained on 20 epochs
history_flow20 = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), steps_per_epoch=len(X_train) / 32, epochs=20, verbose=1, validation_data=(X_test, Y_test), validation_steps=len(X_test) / 32)
history_flow3 = model2.fit_generator(datagen2.flow(X_train, Y_train, batch_size=32), steps_per_epoch=len(X_train) / 32, epochs=20, verbose=1, validation_data=(X_test, Y_test), validation_steps=len(X_test) / 32)

plt.figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
#  "Accuracy"
plt.plot(history_flow20.history['acc'])
plt.plot(history_flow20.history['val_acc'])
plt.plot(history_flow3.history['acc'])
plt.plot(history_flow3.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation', 'train_flow', 'val_flow'], loc='lower right')
plt.show()

plt.figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
# "Loss"
plt.plot(history_flow20.history['loss'])
plt.plot(history_flow20.history['val_loss'])
plt.plot(history_flow3.history['loss'])
plt.plot(history_flow3.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation', 'train_flow', 'val_flow'], loc='upper right')
plt.show()
