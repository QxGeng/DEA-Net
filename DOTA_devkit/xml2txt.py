import math
import os
import xml.etree.cElementTree as ET
import numpy as np

def coordinate_convert_r(box):
    w, h = box[2:-1]
    theta = -box[-1]
    x_lu, y_lu = -w/2, h/2
    x_ru, y_ru = w/2, h/2
    x_ld, y_ld = -w/2, -h/2
    x_rd, y_rd = w/2, -h/2

    x_lu_ = math.cos(theta)*x_lu + math.sin(theta)*y_lu + box[0]
    y_lu_ = -math.sin(theta)*x_lu + math.cos(theta)*y_lu + box[1]

    x_ru_ = math.cos(theta) * x_ru + math.sin(theta) * y_ru + box[0]
    y_ru_ = -math.sin(theta) * x_ru + math.cos(theta) * y_ru + box[1]

    x_ld_ = math.cos(theta) * x_ld + math.sin(theta) * y_ld + box[0]
    y_ld_ = -math.sin(theta) * x_ld + math.cos(theta) * y_ld + box[1]

    x_rd_ = math.cos(theta) * x_rd + math.sin(theta) * y_rd + box[0]
    y_rd_ = -math.sin(theta) * x_rd + math.cos(theta) * y_rd + box[1]

    convert_box = [x_lu_, y_lu_, x_ru_, y_ru_, x_rd_, y_rd_, x_ld_, y_ld_]

    return convert_box

def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 9],
           and has [x1, y1, x2, y2, x3, y3, x4, y4, label] in a per row
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        # ship
        if child_of_root.tag == 'Img_SizeWidth':
            img_width = int(child_of_root.text)
        if child_of_root.tag == 'Img_SizeHeight':
            img_height = int(child_of_root.text)
        if child_of_root.tag == 'HRSC_Objects':
            box_list = []
            for child_item in child_of_root:
                if child_item.tag == 'HRSC_Object':
                    label = 1
                    # for child_object in child_item:
                    #     if child_object.tag == 'Class_ID':
                    #         label = NAME_LABEL_MAP[child_object.text]
                    tmp_box = [0., 0., 0., 0., 0.]
                    for node in child_item:
                        if node.tag == 'mbox_cx':
                            tmp_box[0] = float(node.text)
                        if node.tag == 'mbox_cy':
                            tmp_box[1] = float(node.text)
                        if node.tag == 'mbox_w':
                            tmp_box[2] = float(node.text)
                        if node.tag == 'mbox_h':
                            tmp_box[3] = float(node.text)
                        if node.tag == 'mbox_ang':
                            tmp_box[4] = float(node.text)

                    tmp_box = coordinate_convert_r(tmp_box)
                        # assert label is not None, 'label is none, error'
                    # tmp_box.append(label)
                    tmp_box.append('ship')
                    for node in child_item:
                        if node.tag == 'difficult':
                            tmp_box.append(int(node.text))
                    # if len(tmp_box) != 0:
                    box_list.append(tmp_box)
            # box_list = coordinate_convert(box_list)
            # print(box_list)
    # gtbox_label = np.array(box_list, dtype=np.int32)
    gtbox_label = box_list

    return img_height, img_width, gtbox_label



def WriteTxtFiles(filename, xml_path, gtbox_label_list, output_path):
    filename_txt = filename.split('.')[0]
    box = gtbox_label_list
    outfile = open(output_path + filename_txt + '.txt', 'w')
    outfile.write("imagesource:GooleEarth" + "\n")
    outfile.write("gsd:null" + "\n")
    if len(box) > 0:
        for i in range(len(box)):
            outfile.write(str(box[i][0]) + ' ' + str(box[i][1]) + ' '+ str(box[i][2])+ ' '+ str(box[i][3]) + ' '+ str(box[i][4]) + ' '
                          + str(box[i][5]) + ' '+ str(box[i][6])+ ' ' + str(box[i][7]) + ' '+ str(box[i][8])+ ' ' + str(box[i][9]) + '\n')
            # outfile.write('\n')
    outfile.close()
    # print(type(gtbox_label_list))
    # print(len(gtbox_label_list))
    # print(gtbox_label_list[0][0])

if __name__ == '__main__':
        src_xml_path = '/home/gqx/GQX/AerialDetection/data/HRSC2016/Test/Annotations'

        # xml_path = '/home/gqx/GQX/AerialDetection/data/HRSC2016/Train/Annotations'
        output_path = '/home/gqx/GQX/AerialDetection/data/HRSC2016txt/Test/labeltxt/'

        src_xmls = os.listdir(src_xml_path)

        for x in src_xmls:
            x_path = os.path.join(src_xml_path, x)
            # print(x.split('.')[0])

            img_height, img_width, gtbox_label = read_xml_gtbox_and_label(x_path)

            WriteTxtFiles(x, src_xml_path, gtbox_label, output_path)