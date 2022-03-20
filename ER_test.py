from ER import *
import random


# 【主泵-磨损】 测试数据集
def dataset_main_pump_friction():
    sign_name = ['1909A4C4CBBF754981C2559C35182A2A-低1报警/超差',
                 '542C732FB8B14740958F60E4BB73C047-单调缓慢增长',
                 '3301152A5CA16C418DD797759C2F3CA2-高1报警/超差',
                 'A4DC319E8D388E4EA4893EC08BD841BF-高1报警/超差',
                 '135AF37FD54CFB47B5FDF5CF24836B69-高1报警/超差',
                 'C78C706D46B1EC4E8702397F81A7D27F-低1报警/超差',
                 'A145EDA043224B47B80098DDF5DA4787-低1报警/超差',
                 '18E38A88738240468EDE9EECA66B0732-低1报警/超差',
                 '6E6685BFD6485640AF2A360606839A38-高1报警/超差',
                 '2983D94D05AD344BB74D0B203501A6AA-单调缓慢下降',
                 '3A30244B0FF2BB4FB5A4A8CC22975C44-单调缓慢下降'
                 ]
    whether_reason_sign = [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]
    whether_satisfy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    weight_table = [['规则', '1909A4C4CBBF754981C2559C35182A2A-低1报警/超差', '542C732FB8B14740958F60E4BB73C047-单调缓慢增长', '3301152A5CA16C418DD797759C2F3CA2-高1报警/超差','A4DC319E8D388E4EA4893EC08BD841BF-高1报警/超差', '135AF37FD54CFB47B5FDF5CF24836B69-高1报警/超差', 'C78C706D46B1EC4E8702397F81A7D27F-低1报警/超差', 'A145EDA043224B47B80098DDF5DA4787-低1报警/超差', '18E38A88738240468EDE9EECA66B0732-低1报警/超差', '6E6685BFD6485640AF2A360606839A38-高1报警/超差', '2983D94D05AD344BB74D0B203501A6AA-单调缓慢下降', '3A30244B0FF2BB4FB5A4A8CC22975C44-单调缓慢下降', '失效模式'],
     ['主泵-机封-三级机封-动环-磨损-规则12', '低(0-0.5]', '低(0-0.5]', '高(0.5-1)', '高(0.5-1)', '低(0-0.5]', '高(0.5-1)', '高(0.5-1)', '0', '0', '0', '0', 'F001-磨损'],
     ['主泵-机封-三级机封-动环-磨损-规则11', '低(0-0.5]', '低(0-0.5]', '高(0.5-1)', '高(0.5-1)', '0', '高(0.5-1)', '高(0.5-1)', '低(0-0.5]', '0', '0', '0', 'F001-磨损'],
     ['主泵-机封-三级机封-动环-磨损-规则10', '低(0-0.5]', '低(0-0.5]', '高(0.5-1)', '高(0.5-1)', '0', '高(0.5-1)', '高(0.5-1)', '0', '低(0-0.5]', '0', '0', 'F001-磨损'],
     ['主泵-机封-三级机封-动环-磨损-规则09', '低(0-0.5]', '低(0-0.5]', '高(0.5-1)', '高(0.5-1)', '低(0-0.5]', '0', '高(0.5-1)', '0', '0', '高(0.5-1)', '0', 'F001-磨损'],
     ['主泵-机封-三级机封-动环-磨损-规则08', '低(0-0.5]', '低(0-0.5]', '高(0.5-1)', '高(0.5-1)', '0', '0', '高(0.5-1)', '低(0-0.5]', '0', '高(0.5-1)', '0', 'F001-磨损'],
     ['主泵-机封-三级机封-动环-磨损-规则07', '低(0-0.5]', '低(0-0.5]', '高(0.5-1)', '高(0.5-1)', '0', '0', '高(0.5-1)', '0', '低(0-0.5]', '高(0.5-1)', '0', 'F001-磨损'],
     ['主泵-机封-三级机封-动环-磨损-规则06', '低(0-0.5]', '低(0-0.5]', '高(0.5-1)', '高(0.5-1)', '低(0-0.5]', '高(0.5-1)', '0', '0', '0', '0', '高(0.5-1)', 'F001-磨损'],
     ['主泵-机封-三级机封-动环-磨损-规则05', '低(0-0.5]', '低(0-0.5]', '高(0.5-1)', '高(0.5-1)', '0', '高(0.5-1)', '0', '低(0-0.5]', '0', '0', '高(0.5-1)', 'F001-磨损'],
     ['主泵-机封-三级机封-动环-磨损-规则04', '低(0-0.5]', '低(0-0.5]', '高(0.5-1)', '高(0.5-1)', '0', '高(0.5-1)', '0', '0', '低(0-0.5]', '0', '高(0.5-1)', 'F001-磨损'],
     ['主泵-机封-三级机封-动环-磨损-规则03', '低(0-0.5]', '低(0-0.5]', '高(0.5-1)', '高(0.5-1)', '低(0-0.5]', '0', '0', '0', '0', '高(0.5-1)', '高(0.5-1)', 'F001-磨损'],
     ['主泵-机封-三级机封-动环-磨损-规则02', '低(0-0.5]', '低(0-0.5]', '高(0.5-1)', '高(0.5-1)', '0', '0', '0', '低(0-0.5]', '0', '高(0.5-1)', '高(0.5-1)', 'F001-磨损'],
     ['主泵-机封-三级机封-动环-磨损-规则01', '低(0-0.5]', '低(0-0.5]', '高(0.5-1)', '高(0.5-1)', '0', '0', '0', '0', '低(0-0.5]', '高(0.5-1)', '高(0.5-1)', 'F001-磨损']]
    return weight_table, sign_name, whether_reason_sign, whether_satisfy


def test_results_type():
    weight_table, sign_name, whether_reason_sign, whether_satisfy = dataset_main_pump_friction()
    assert len(sign_name) == len(whether_reason_sign) == len(whether_satisfy)
    weight_table = np.array(weight_table)
    whether_satisfy = np.array(whether_satisfy)
    calculated_rules, rule2failuremode, failuremodes, failuremodes_alarm_map = methondcal(weight_table, sign_name,
                                                                                          whether_reason_sign,
                                                                                          whether_satisfy)
    # print("#######################results############################")
    # print(calculated_rules)
    # print(rule2failuremode)
    # print(failuremodes)
    # print(failuremodes_alarm_map)
    assert type(calculated_rules) == type({"name": "xi"})
    assert type(rule2failuremode) == type({"name": "xi"})

    for key in calculated_rules.keys():
        assert type(calculated_rules[key]) == type(1.0)
    for key in failuremodes.keys():
        assert type(failuremodes[key]) == type(1.0)
    for key in failuremodes_alarm_map.keys():
        assert failuremodes_alarm_map[key] in ['A', 'B', 'C', 'D']


# 共产生9种组合:
# [1]: 都不满足
def test_no_satisfy():
    # 读取数据
    weight_table, sign_name, whether_reason_sign, whether_satisfy = dataset_main_pump_friction()
    assert len(sign_name) == len(whether_reason_sign) == len(whether_satisfy)

    for i in range(len(whether_satisfy)):
        whether_satisfy[i] = 0
    weight_table = np.array(weight_table)
    whether_satisfy = np.array(whether_satisfy)
    calculated_rules, rule2failuremode, failuremodes, failuremodes_alarm_map = methondcal(weight_table, sign_name, whether_reason_sign, whether_satisfy)
    # 由于征兆都不满足，因此输出必为0
    for key in calculated_rules.keys():
        assert abs(calculated_rules[key]-0) < 1e-5

    for key in failuremodes.keys():
        assert abs(failuremodes[key]-0) < 1e-5

    for key in failuremodes_alarm_map.keys():
        assert failuremodes_alarm_map[key] == 'A'


# 部分满足，都不满足
def test_part_satisfy_trend_threshold_no_reason():
    # 读取数据
    weight_table, sign_name, whether_reason_sign, whether_satisfy = dataset_main_pump_friction()
    assert len(sign_name) == len(whether_reason_sign) == len(whether_satisfy)
    not_reason_sign_index = []
    for i in range(len(whether_reason_sign)):
        if not whether_reason_sign[i]:
            not_reason_sign_index.append(i)
    random.shuffle(not_reason_sign_index)
    # 选其中的一半满足
    for i in range(len(not_reason_sign_index) // 2):
        whether_satisfy[not_reason_sign_index[i]] = 1

    weight_table = np.array(weight_table)
    whether_satisfy = np.array(whether_satisfy)
    calculated_rules, rule2failuremode, failuremodes, failuremodes_alarm_map = methondcal(weight_table, sign_name, whether_reason_sign, whether_satisfy)
    # 由于征兆都不满足，因此输出必为0
    for key in calculated_rules.keys():
        assert abs(calculated_rules[key]-0) < 1e-5

    for key in failuremodes.keys():
        assert abs(failuremodes[key]-0) < 1e-5

    for key in failuremodes_alarm_map.keys():
        assert failuremodes_alarm_map[key] == 'A'


# 全满足，都不满足
def test_all_satisfy_trend_threshold_no_reason():
    # 读取数据
    weight_table, sign_name, whether_reason_sign, whether_satisfy = dataset_main_pump_friction()
    assert len(sign_name) == len(whether_reason_sign) == len(whether_satisfy)
    for i in range(len(whether_reason_sign)):
        if not whether_reason_sign[i]:
            whether_satisfy[i] = 1

    weight_table = np.array(weight_table)
    whether_satisfy = np.array(whether_satisfy)
    calculated_rules, rule2failuremode, failuremodes, failuremodes_alarm_map = methondcal(weight_table, sign_name, whether_reason_sign, whether_satisfy)
    # 由于征兆都不满足，因此输出必为0
    for key in calculated_rules.keys():
        assert calculated_rules[key]-0 < 1 and calculated_rules[key] > 0

    for key in failuremodes.keys():
        assert abs(failuremodes[key]) < 1 and failuremodes[key] > 0

    for key in failuremodes_alarm_map.keys():
        assert failuremodes_alarm_map[key] == 'C'


# 都不满足，部分满足
def test_no_satisfy_trend_threshold_part_reason():
    # 读取数据
    weight_table, sign_name, whether_reason_sign, whether_satisfy = dataset_main_pump_friction()
    reason_signs_index = []
    for i in range(len(whether_reason_sign)):
        if whether_reason_sign[i]:
            reason_signs_index.append(i)
    random.shuffle(reason_signs_index)
    for i in range(min(len(reason_signs_index)//2+1, len(reason_signs_index))):
        whether_satisfy[reason_signs_index[i]] = 1

    weight_table = np.array(weight_table)
    whether_satisfy = np.array(whether_satisfy)
    calculated_rules, rule2failuremode, failuremodes, failuremodes_alarm_map = methondcal(weight_table, sign_name, whether_reason_sign, whether_satisfy)
    # 由于征兆都不满足，因此输出必为0
    for key in calculated_rules.keys():
        assert abs(calculated_rules[key] - 0) < 1e-5

    for key in failuremodes.keys():
        assert abs(failuremodes[key]) < 1e-5

    for key in failuremodes_alarm_map.keys():
        assert failuremodes_alarm_map[key] == 'B'


# 部门满足，部分满足
def test_part_satisfy_trend_threshold_part_reason():
    # 读取数据
    weight_table, sign_name, whether_reason_sign, whether_satisfy = dataset_main_pump_friction()
    reason_signs_index = []
    not_reason_signs_index = []
    for i in range(len(whether_reason_sign)):
        if whether_reason_sign[i]:
            reason_signs_index.append(i)
        else:
            not_reason_signs_index.append(i)
    random.shuffle(reason_signs_index)
    for i in range(min(len(reason_signs_index)//2+1, len(reason_signs_index))):
        whether_satisfy[reason_signs_index[i]] = 1

    random.shuffle(not_reason_signs_index)
    for i in range(len(not_reason_signs_index)//2):
        whether_satisfy[not_reason_signs_index[i]] = 1

    weight_table = np.array(weight_table)
    whether_satisfy = np.array(whether_satisfy)
    calculated_rules, rule2failuremode, failuremodes, failuremodes_alarm_map = methondcal(weight_table, sign_name, whether_reason_sign, whether_satisfy)
    # 由于征兆都不满足，因此输出必为0
    for key in calculated_rules.keys():
        assert abs(calculated_rules[key] - 0) < 1e-5

    for key in failuremodes.keys():
        assert abs(failuremodes[key]) < 1e-5

    for key in failuremodes_alarm_map.keys():
        assert failuremodes_alarm_map[key] == 'B'


# 全满足，部分满足
def test_all_satisfy_trend_threshold_part_reason():
    # 读取数据
    weight_table, sign_name, whether_reason_sign, whether_satisfy = dataset_main_pump_friction()
    reason_signs_index = []
    for i in range(len(whether_reason_sign)):
        if whether_reason_sign[i]:
            reason_signs_index.append(i)
        else:
            whether_satisfy[i] = 1
    random.shuffle(reason_signs_index)
    for i in range(min(len(reason_signs_index)//2+1, len(reason_signs_index))):
        whether_satisfy[reason_signs_index[i]] = 1

    weight_table = np.array(weight_table)
    whether_satisfy = np.array(whether_satisfy)
    calculated_rules, rule2failuremode, failuremodes, failuremodes_alarm_map = methondcal(weight_table, sign_name, whether_reason_sign, whether_satisfy)
    # 由于征兆都不满足，因此输出必为0
    for key in calculated_rules.keys():
        assert calculated_rules[key] > 0

    for key in failuremodes.keys():
        assert failuremodes[key] > 0

    for key in failuremodes_alarm_map.keys():
        if min(len(reason_signs_index)//2+1, len(reason_signs_index)) >= 1:
            assert failuremodes_alarm_map[key] == 'D'
        else:
            assert failuremodes_alarm_map[key] == 'C'


# 都不满足，全满足
def test_no_satisfy_trend_threshold_all_reason():
    # 读取数据
    weight_table, sign_name, whether_reason_sign, whether_satisfy = dataset_main_pump_friction()
    for i in range(len(whether_reason_sign)):
        if whether_reason_sign[i]:
            whether_satisfy[i] = 1

    weight_table = np.array(weight_table)
    whether_satisfy = np.array(whether_satisfy)
    calculated_rules, rule2failuremode, failuremodes, failuremodes_alarm_map = methondcal(weight_table, sign_name, whether_reason_sign, whether_satisfy)
    # 由于征兆都不满足，因此输出必为0
    for key in calculated_rules.keys():
        assert abs(calculated_rules[key] - 0) < 1e-5

    for key in failuremodes.keys():
        assert abs(failuremodes[key]) < 1e-5

    for key in failuremodes_alarm_map.keys():
        assert failuremodes_alarm_map[key] == 'B'


# 部分满足，全满足
def test_part_satisfy_trend_threshold_all_reason():
    # 读取数据
    weight_table, sign_name, whether_reason_sign, whether_satisfy = dataset_main_pump_friction()
    assert len(sign_name) == len(whether_reason_sign) == len(whether_satisfy)
    not_reason_sign_index = []
    for i in range(len(whether_reason_sign)):
        if not whether_reason_sign[i]:
            not_reason_sign_index.append(i)
        else:
            whether_satisfy[i] = 1
    random.shuffle(not_reason_sign_index)
    # 选其中的一半满足
    for i in range(len(not_reason_sign_index) // 2):
        whether_satisfy[not_reason_sign_index[i]] = 1

    weight_table = np.array(weight_table)
    whether_satisfy = np.array(whether_satisfy)
    calculated_rules, rule2failuremode, failuremodes, failuremodes_alarm_map = methondcal(weight_table, sign_name, whether_reason_sign, whether_satisfy)
    # 由于征兆都不满足，因此输出必为0
    for key in calculated_rules.keys():
        assert abs(calculated_rules[key]-0) < 1e-5

    for key in failuremodes.keys():
        assert abs(failuremodes[key]-0) < 1e-5

    for key in failuremodes_alarm_map.keys():
        assert failuremodes_alarm_map[key] == 'B'


# 全满足，全满足
def test_all_satisfy_trend_threshold_all_reason():
    # 读取数据
    weight_table, sign_name, whether_reason_sign, whether_satisfy = dataset_main_pump_friction()
    assert len(sign_name) == len(whether_reason_sign) == len(whether_satisfy)

    for i in range(len(whether_satisfy)):
        whether_satisfy[i] = 1
    weight_table = np.array(weight_table)
    whether_satisfy = np.array(whether_satisfy)
    calculated_rules, rule2failuremode, failuremodes, failuremodes_alarm_map = methondcal(weight_table, sign_name, whether_reason_sign, whether_satisfy)
    # 由于征兆都不满足，因此输出必为0
    for key in calculated_rules.keys():
        assert calculated_rules[key]-0 > 0

    for key in failuremodes.keys():
        assert failuremodes[key]-0 > 0

    for key in failuremodes_alarm_map.keys():
        assert failuremodes_alarm_map[key] == 'D'


if __name__ == "__main__":
    pass