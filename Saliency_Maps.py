    # Saliency map visualization
    # To generate a stay-green or rapid senescence example, change activ_index to either 230 or 190, respectively.
    # To produce the activation maps, paste this portion of code immediately after the eval_list[i], metrics_list[i],
    #  and labels_list[i] objects (approx. line 464). 
    
    # Example of stay-green: 230 
    # Example of rapid senescence: 190

    activ_index = 197
    sample_images = testx_4d[activ_index]
    print(list(testx_dict.keys())[activ_index])
    check_geno = list(testx_dict.keys())[activ_index]
    print(testy[activ_index])
    print(scores_clean_subset.loc[scores_clean_subset['Pltg_ID_Key_JPG'] == check_geno, traitname])
    
    sample_images.shape
    sample_images_tensor = tf.convert_to_tensor(sample_images, dtype=tf.float32)
    sample_images_tensor = tf.expand_dims(sample_images_tensor, axis=0)  
    
    layer_name = 'dense_1' # Note that this should refer to the second (of two) dense layers. The name of each layer can be obtained by bestmodel.summary()
    layer_idx = [idx for idx, layer in enumerate(bestmodel.layers) if layer.name == layer_name][0]
    bestmodel.layers[layer_idx].activation = tf.keras.activations.linear
    bestmodel = tf.keras.models.Model(inputs=bestmodel.inputs, outputs=bestmodel.layers[layer_idx].output)
    bestmodel.compile()
    
    # Compute the saliency map using GradientTape
    with tf.GradientTape() as tape:
        tape.watch(sample_images_tensor)
        predictions = bestmodel(sample_images_tensor)
    
    grads = tape.gradient(predictions, sample_images_tensor)[0]
    saliency_map = np.sum(np.abs(grads), axis=-1)
    saliency_map = ndimage.gaussian_filter(saliency_map, sigma=5)
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))
    
    
    num_time_points = 14
    channel_per_time_point = 3

    # Reshape the image to separate time points
    img_time_points = sample_images.reshape(163, 163, num_time_points, channel_per_time_point)

    for time_point_index in range(num_time_points):
        img_to_display = img_time_points[:, :, time_point_index, :]
    
        # Display original image and saliency map
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f'Original Image Flight {time_point_index + 1}/14')
        plt.imshow(img_to_display)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(f'Saliency Map Flight {time_point_index + 1}/14')
        plt.imshow(img_to_display)
        plt.imshow(saliency_map, cmap='jet', alpha=0.5)  # Adjust alpha to tune transparency
        plt.axis('off')
    
        plt.savefig(savedir + f'{time_point_index}.jpeg', dpi=600, bbox_inches='tight')
        plt.close()