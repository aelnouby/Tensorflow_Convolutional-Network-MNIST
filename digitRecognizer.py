import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Paramters
num_lables=10
image_size=28

learning_rate=1e-2
batch_size=50

#Read Data
data=pd.read_csv('train.csv',header=0)

test=pd.read_csv('test.csv',header=0)


images=np.array(data.drop('label',axis=1)).astype(np.float32)
images=np.multiply(images,1.0/255.0)
labels=np.array(data.label).astype(np.float32)

labels=(np.arange(10)==labels[:,None]).astype(np.float32)


# def display(img):
#     dis=img.reshape(image_size,image_size)
#     plt.imshow(dis,cmap=cm.binary)
#
# print(display(images[22]))

#Splitting into training and validation
validation_size=2000

training_data=images[validation_size:]
training_labels=labels[validation_size:]

validation_data=images[:validation_size]
validation_labels=labels[:validation_size]

#helper methods
def weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def biases(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))

def conv2d(data,wts):
    return tf.nn.conv2d(data,wts,strides=[1,1,1,1],padding='SAME',use_cudnn_on_gpu=False)

def max_pooling(value):
    return tf.nn.max_pool(value,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')






graph=tf.Graph()
with graph.as_default():


    tf_train_data=tf.placeholder(tf.float32,shape=[batch_size,image_size*image_size])
    tf_train_labels=tf.placeholder(tf.float32,shape=[batch_size,num_lables])

    tf_valid_data=tf.constant(validation_data)
    tf_test_data=tf.placeholder(tf.float32,shape=[batch_size,image_size*image_size])
    # tf_valid_labels=tf.placeholder(tf.float32,shape=[batch_size,num_lables])

    layer1_w=weights([5,5,1,32])
    layer1_b=biases([32])

    #Layer2
    layer2_w=weights([5,5,32,64])
    layer2_b=biases([64])

    #fully connected
    fully_w=weights([7*7*64,1024])
    fully_b=biases([1024])
    final_w=weights([1024,num_lables])
    final_b=biases([num_lables])


    def model(data):

        image=tf.reshape(data,[-1,image_size,image_size,1])

        hidden1_conv=tf.nn.relu(conv2d(image,layer1_w)+layer1_b)
        hidden1_pool=max_pooling(hidden1_conv)

        hidden2_conv=tf.nn.relu(conv2d(hidden1_pool,layer2_w)+layer2_b)
        hidden2_pool=max_pooling(hidden2_conv)

        hidden2_pool_flat=tf.reshape(hidden2_pool,[-1,7*7*64])
        hidden_fully=tf.nn.relu(tf.matmul(hidden2_pool_flat,fully_w)+fully_b)

        #dropout
        hidden_final=tf.nn.dropout(hidden_fully,keep_prob=1)


        return  tf.matmul(hidden_final,final_w)+final_b


    logits=model(tf_train_data)
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,tf_train_labels))

    optimization=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    train_predections=tf.nn.softmax(logits)
    valid_predections=tf.nn.softmax(model(tf_valid_data))
    test_prediction=tf.nn.softmax(model(tf_test_data))


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])



test_data=np.array(pd.read_csv('test.csv',header =0)).astype(np.float32)
test_data=np.multiply(test_data,1.0/255.0)

predicted_lables = np.zeros(test_data.shape[0])

num_steps=20001

with tf.Session(graph=graph) as session:

    tf.initialize_all_variables().run()
    print('initialized')

    counter=0
    for step in range(num_steps):
        if counter>len(training_data):
            prem=np.arange(len(training_data))
            np.random.shuffle(prem)
            training_data=training_data[prem]
            training_labels=training_labels[perm]
            counter=0
        offset = (step * batch_size) % (training_labels.shape[0] - batch_size)
        batch_data = training_data[offset:(offset + batch_size), :]
        batch_labels = training_labels[offset:(offset + batch_size), :]

        feed_dict={tf_train_data:batch_data,tf_train_labels:batch_labels}

        _,l, predictions = session.run([optimization, loss, train_predections], feed_dict=feed_dict)

        if(step%500==0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_predections.eval(), validation_labels))

        counter+=1

    # test_rslt=test_prediction.eval()
    for i in range(0,test_data.shape[0]//batch_size):
    	rslt=test_prediction.eval(feed_dict={tf_test_data: test_data[i*batch_size : (i+1)*batch_size]})
    	predicted_lables[i*batch_size : (i+1)*batch_size] = np.argmax(rslt,1)

     	









np.savetxt('submission_softmax.csv',
           np.c_[range(1,len(predicted_lables)+1),predicted_lables],
           delimiter=',',
           header = 'ImageId,Label',
           comments = '',
           fmt='%d')






