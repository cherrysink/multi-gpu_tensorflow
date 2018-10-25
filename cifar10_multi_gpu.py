import os.path
import re
import time
import tensorflow as tf
import cifar10

data_dir = './tem/cifar10_data/cifar-10-batches-bin'

batch_size=128
max_steps=1000000

num_gpus=2


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# 不同gpu的loss合在一起
def tower_loss(scope,images,labels):

    logits=cifar10.inference(images)
    _=cifar10.loss(logits,labels)
    lossses=tf.get_collection('losses',scope)
    total_loss=tf.add_n(lossses,name='total_loss')
    return total_loss
# 不同的gpu梯度合成
def average_gradients(tower_grads):
    average_grads=[]
    for grad_and_vars in zip(*tower_grads):
        grads=[]
        for g,_ in grad_and_vars:
            expanded_g=tf.expand_dims(g,0)
            grads.append(expanded_g)
        grad=tf.concat(axis=0,values=grads)
        grad=tf.reduce_mean(grad,0)
        v=grad_and_vars[0][1]
        grad_and_vars=(grad,v)
        average_grads.append(grad_and_vars)
    return average_grads

# 训练函数
def train():
    with tf.Graph().as_default(),tf.device('/cpu:0'):
        global_step=tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)
        num_batches_per_epoch=cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/batch_size
        decay_steps=int(num_batches_per_epoch*cifar10.NUM_EPOCHS_PER_DECAY)
        lr=tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,global_step,decay_steps,cifar10.LEARNING_RATE_DECAY_FACTOR,staircase=True)
        opt=tf.train.GradientDescentOptimizer(lr)

        images, labels = cifar10.distorted_inputs()
        # 使用预加载的队列,使用这个会速度可以提高五倍，亲测
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=2 * FLAGS.num_gpus)
        tower_grads=[]
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME,i)) as scope:
                    loss=tower_loss(scope,images,labels)
                    tf.get_variable_scope().reuse_variables()
                    grads=opt.compute_gradients(loss)
                    tower_grads.append(grads)

        grads=average_gradients(tower_grads)
        apply_gradient_op=opt.apply_gradients(grads,global_step=global_step)

        saver=tf.train.Saver(tf.all_variables())
        init=tf.global_variables_initializer()
        sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        for step in range(max_steps):
            start_time=time.time()
            _,loss_value=sess.run([apply_gradient_op,loss])
            duration=time.time()-start_time

            if step%10==0:
                num_examples_per_step=batch_size*num_gpus
                examples_per_sec=num_examples_per_step /duration
                sec_per_batch=duration/num_gpus

                format_str=('step %d, loss= %.2f (%.1f example/sec; %.3f' 'sec/batch)')
                print(format_str % (step,loss_value,examples_per_sec,sec_per_batch))

                if step%1000==0 or (step+1)==max_steps:
                    saver.save(sess,'/',global_step=step)
# cifar10.maybe_download_and_extract()

if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
tf.gfile.MakeDirs(FLAGS.train_dir)
train()