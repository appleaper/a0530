import choise

model_dict = {
    'gener_classif' : choise.gener_train,
    'fanpai_classif' : choise.fanpai_train,
}

if __name__ == '__main__':
    # model_dict['gener_classif']()
    model_dict['fanpai_classif']()