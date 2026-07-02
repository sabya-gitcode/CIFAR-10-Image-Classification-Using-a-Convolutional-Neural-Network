
import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train.shape, y_train.shape, x_test.shape, y_test.shape

model = keras.Sequential([
    keras.Input(shape=(32,32,3)),
    keras.layers.Conv2D(filters = 32,
    kernel_size = (3, 3),
    padding = 'same',
    strides = (1,1),
    activation ='relu',
    kernel_regularizer = 'l2'),

    keras.layers.Conv2D(filters = 64,
    kernel_size = (3, 3),
    padding = 'same',
    strides = (1,1),
    activation ='relu',
    kernel_regularizer = 'l2'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),


    keras.layers.Conv2D(filters = 128,
    kernel_size = (3, 3),
    padding = 'same',
    strides = (1,1),
    activation ='relu',
    kernel_regularizer = 'l2'),

    keras.layers.Conv2D(filters = 256,
    kernel_size = (3, 3),
    padding = 'same',
    strides = (1,1),
    activation ='relu',
    kernel_regularizer = 'l2'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),

    keras.layers.Conv2D(filters = 512,
    kernel_size = (3, 3),
    padding = 'same',
    strides = (1,1),
    activation ='relu',
    kernel_regularizer = 'l2'),

    keras.layers.Conv2D(filters = 1024,
    kernel_size = (3, 3),
    padding = 'same',
    strides = (1,1),
    activation ='relu',
    kernel_regularizer = 'l2'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),

    keras.layers.Flatten(),
    keras.layers.Dense(64, activation = 'relu', kernel_regularizer = 'l2'),
    keras.layers.Dropout(0.5),


    keras.layers.Dense(32, activation = 'relu', kernel_regularizer = 'l2'),
    keras.layers.Dropout(0.5),


    keras.layers.Dense(10, activation = 'softmax'),
])
model.summary()

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Early stopping

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=0,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0,
)
early_stop

history = model.fit(
    x = x_train,
    y = y_train,
    epochs = 10,
    batch_size = 32,
    validation_split = 0.2,
    callbacks = [early_stop]
)

model.evaluate(x_test, y_test)

model.save("cifar10_cnn.keras")