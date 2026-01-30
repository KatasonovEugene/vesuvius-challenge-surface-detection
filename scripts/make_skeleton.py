from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation

def generate_tubed_skeleton_numpy(label_vol):
    mask = (label_vol == 1)
    skel = skeletonize(mask)
    tubed_skel = binary_dilation(skel, iterations=1)
    return tubed_skel.astype(np.float32)[..., None]

def add_skeleton_target(image, label):
    tubed_skel = tf.numpy_function(
        func=generate_tubed_skeleton_numpy,
        inp=[label],
        Tout=tf.float32
    )
    tubed_skel.set_shape(label.shape)
    combined_label = tf.concat([tf.cast(label, tf.float32), tubed_skel], axis=-1)
    return image, combined_label

def run():
    train_images
    pass

if __name__ == "__main__":
    run()