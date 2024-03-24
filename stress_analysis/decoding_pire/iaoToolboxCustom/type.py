from iao_toolbox import helpers


class Subject:

    def __init__(self, name):
        self.bioData = []
        self.bioList = []
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def add_data_list(self, root, sub, bio, opt='', ending=None, sort=False):
        raw_path = helpers.get_raw_path(root, (sub, bio, *opt))
        self.bioData.append(helpers.get_file_list(raw_path, ending, True, sort))
        self.bioList.append(bio)
