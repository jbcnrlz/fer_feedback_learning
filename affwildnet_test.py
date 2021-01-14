import affwildnet.vggface_gru_tf2 as net
import cv2, os
import tensorflow as tf

def getImage():
    returnData = []
    for i in range(1,81):
        imOpened = cv2.imread(os.path.join('frames_face','roi_0_frame_%d.jpg' % i))
        imOpened = cv2.resize(imOpened,(96,96))
        returnData.append(imOpened)

    tensor = tf.convert_to_tensor([returnData],dtype=tf.float32)
    tensor -= 128.0
    tensor /= 128.0
    return tensor

def main():
    imagesToConv = getImage()
    imagesToConv = tf.reshape(imagesToConv, [-1, 96, 96, 3])

    network = net.VGGFace(1,80)
    #slim = tf.slim
    network.setup(imagesToConv)
    prediction = network.get_output()
    variables_to_restore = tf.compat.v1.global_variables()
    valence_val = prediction[:,0]
    arousal_val = prediction[:,1]
    print(valence_val)
    print(arousal_val)
    restor = tf.compat.v1.train.Saver(var_list=variables_to_restore)
    with tf.compat.v1.Session() as sess:
        restor.restore(sess,'/home/joaocardia/PycharmProjects/fer_feedback_learning/wtgs_affwildnet/model.ckpt-0')
        a = sess.run(imagesToConv)
        valence_val = a[:, 0]
        arousal_val = a[:, 1]
        print(valence_val)
        print(arousal_val)

    print('po')


if __name__ == '__main__':
    main()