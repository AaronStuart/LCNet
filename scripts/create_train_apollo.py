import os
apollo_root = '/media/stuart/data/dataset/Apollo/Lane_Detection'
image_dirs = [
    '/media/stuart/data/dataset/Apollo/Lane_Detection/ColorImage_road02/ColorImage',
    '/media/stuart/data/dataset/Apollo/Lane_Detection/ColorImage_road03/ColorImage',
    '/media/stuart/data/dataset/Apollo/Lane_Detection/ColorImage_road04/ColorImage'
]
label_dirs = [
    '/media/stuart/data/dataset/Apollo/Lane_Detection/Labels_road02/Label',
    '/media/stuart/data/dataset/Apollo/Lane_Detection/Labels_road03/Label',
    '/media/stuart/data/dataset/Apollo/Lane_Detection/Labels_road04/Label'
]

if __name__ == '__main__':
    images_path = []
    file = open('train_apollo.txt', 'w')
    for image_dir in image_dirs:
        for dirpath, dirnames, filenames in os.walk(image_dir):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    images_path.append(os.path.join(dirpath, filename))

    for image_path in images_path:
        if image_path.find('road02') != -1:
            label_path = image_path.replace('ColorImage_road02/ColorImage', 'Labels_road02/Label').replace('.jpg', '_bin.png')
        if image_path.find('road03') != -1:
            label_path = image_path.replace('ColorImage_road03/ColorImage', 'Labels_road03/Label').replace('.jpg', '_bin.png')
        if image_path.find('road04') != -1:
            label_path = image_path.replace('ColorImage_road04/ColorImage', 'Labels_road04/Label').replace('.jpg', '_bin.png')
        file.writelines(image_path + ',' + label_path + '\n')

    file.close()