from data.base_dataset import BaseDataset, get_params, get_transform
from data.relighting_dataset_single_image import read_component 

def get_data_beta(file_name_input, file_name_output, dataroot, img_transform, multiple_replace_image):

    data = {}  # output dictionary

    data['scene_label'] = file_name_input # use metadata in the future?
    data['light_position_color_original'] = None # image_name2light_condition(file_name_input)
    data['light_position_color_new'] = None # image_name2light_condition(file_name_output)

    # Reflectance_output
    data['Reflectance_output'] = None # read_component(dataroot, 'Reflectance', file_name_input, img_transform)
    data['Shading_ori'] = None # read_component(dataroot, 'Shading', file_name_input, img_transform, r_pil=True)
    data['Shading_output'] = None # read_component(dataroot, 'Shading', file_name_output, img_transform, r_pil=True)
    if multiple_replace_image:
        data['Image_input'] = None # torch.mul(data['Reflectance_output'], data['Shading_ori'])
        data['Image_relighted'] = None # torch.mul(data['Reflectance_output'], data['Shading_output'])
    else:
        data['Image_input'] = read_component(dataroot, 'Image', file_name_input, img_transform)
        data['Image_relighted'] = read_component(dataroot, 'Image', file_name_output, img_transform)

    return data


def read_anno(file_name):
    anno_list = []
    with open(file_name, 'r') as f:
        for x in f.readlines():
            x = x.strip('\n')
            anno_list.append(x)
    return anno_list


class RelightingDatasetSingleImageCustom(BaseDataset):
    """A dataset class for relighting dataset.
       This dataset read data image by image (for test).
       Read test pairs from anno files. Each scene only has 2 images, one for input and one for relighting.
    """

    def __init__(self, opt, validation=False):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        if validation:
            anno_file = opt.anno_validation
        else:
            anno_file = opt.anno
        self.pairs_list = read_anno(anno_file)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains
            'Image_input': source_img,
            'light_position_color_new': None,
            'light_position_color_original': None,
            'Reflectance_output': None,
            'Shading_output': None,
            'Shading_ori': None,
            'Image_relighted': source_img,
            'scene_label': descriptions,
        """
        # get parameters
        dataroot = self.dataroot
        img_size = self.opt.img_size
        multiple_replace_image = self.opt.multiple_replace_image
        # get one pair
        pair = self.pairs_list[index].split()
        file_name_input = pair[0]
        file_name_output = pair[0]

        # get the parameters of data augmentation
        transform_params = get_params(self.opt, img_size)
        img_transform = get_transform(self.opt, transform_params)

        data = get_data_beta(file_name_input, file_name_output, dataroot, img_transform, multiple_replace_image)

        return data

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.pairs_list)


