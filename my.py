from classification_util import ClassificationUtil as cu

gildong = cu()
gildong.read('train.csv')
gildong.show()

gildong.heatmap()

c = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt',
     'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']
gildong.ignore_warning()
gildong.run_svm(c, 'price_range')
gildong.run_neighbor_classifier(c, 'price_range', 10)
gildong.run_logistic_regression(c, 'price_range')
gildong.run_decision_tree_classifier(c, 'price_range')

c = ['battery_power', 'int_memory', 'px_height', 'px_width', 'ram']
gildong.run_svm(c, 'price_range')
gildong.run_neighbor_classifier(c, 'price_range', 10)
gildong.run_logistic_regression(c, 'price_range')
gildong.run_decision_tree_classifier(c, 'price_range')