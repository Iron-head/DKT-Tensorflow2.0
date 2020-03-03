import tensorflow as tf
import dpkt
import data_util
import metrics

file = 'skill_builder_data.csv'
optimizer = 'adam'
CSV_Log = "./logs/train.log"
model_path = "./weights/bestmodel"
log_dir = "logs"

dataset, length, nb_features, nb_skills = data_util.load_dataset(file, batch_size=32,
                                                                 shuffle=True)

train_set, test_set, val_set = data_util.split_dataset(dataset=dataset,
                                                       total_size=length,
                                                       test_fraction=0.2,
                                                       val_fraction=0.2)


print('-------compiling---------')
model = dpkt.DKTModel(nb_features=nb_features,
                      nb_skills=nb_skills,
                      hidden_units=128,
                      dropout_rate=0.3)

model.compile(optimizer=optimizer,
              metrics=[
                  metrics.BinaryAccuracy(),
                  metrics.AUC(),
                  metrics.Precision(),
                  metrics.Recall()
              ])

print(model.summary())
print("\nCompiling Done!")

print("_____________\nTraining!__________________")

model.fit(dataset=train_set,
          epochs=50,
          verbose=1,
          validation_data=val_set,
          callback=[
              tf.keras.callbacks.CSVLogger(CSV_Log),
              tf.keras.callbacks.ModelCheckpoint(model_path,
                                                 save_best_only=True,
                                                 save_weights_only=True),
              tf.keras.callbacks.TensorBoard(log_dir=log_dir)
          ])

print("\n___________TRAINING DONE!_______________")


print("\n___________TESTING!_______________")
model.load_weights(model_path)
model.evaluate(dataset=test_set, verbose=1)
print("\n___________TESTING DONE!_______________")
