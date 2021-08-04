import gin
import tensorflow as tf
import logging


@gin.configurable
class Trainer(object):
    def __init__(self, model, dataset, run_paths, total_steps, log_interval, learning_rate):
        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.model = model
        self.dataset = dataset
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval

        # Summary Writer
        self.train_summary_writer = tf.summary.create_file_writer(run_paths['summary'] + "/train")
        self.test_summary_writer = tf.summary.create_file_writer(run_paths['summary'] + "/test")

        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=model)
        self.manager = tf.train.CheckpointManager(self.ckpt, run_paths["path_ckpts_train"], max_to_keep=5)

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def train(self):
        # Restore checkpoint if exists
        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint)
            logging.info("Restored from {}.".format(self.manager.latest_checkpoint))
        else:
            logging.info("Train from scratch.")

        for idx, (images, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(images, labels)

            # template = 'Step {}, Loss: {}, Accuracy: {}'
            # print(template.format(step,
            #                       self.train_loss.result(),
            #                       self.train_accuracy.result() * 100))

            if step % self.log_interval == 0:

                # Reset test metrics
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()

                for test_images, test_labels in self.ds_val:
                    self.test_step(test_images, test_labels)

                template = 'Step {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.test_loss.result(),
                                             self.test_accuracy.result() * 100))

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                # Write summary to tensorboard
                with self.test_summary_writer.as_default():
                    tf.summary.scalar("test_loss", self.test_loss.result(), step=step // self.log_interval)
                    tf.summary.scalar("test_accuracy", self.test_accuracy.result(), step=step // self.log_interval)

                yield self.test_accuracy.result().numpy()

            with self.train_summary_writer.as_default():
                tf.summary.scalar("train_loss", self.train_loss.result(), step=step)
                tf.summary.scalar("train_accuracy", self.train_accuracy.result(), step=step)

            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
                self.manager.save()

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                self.manager.save()
                self.model.save(self.run_paths["saved_model"])
                return self.test_accuracy.result().numpy()
