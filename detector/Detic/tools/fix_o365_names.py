# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", default='datasets/objects365/annotations/zhiyuan_objv2_val.json')
    parser.add_argument("--fix_name_map", default='datasets/metadata/Objects365_names_fix.csv')
    args = parser.parse_args()

    new_names = {}
    old_names = {}
    with open(args.fix_name_map, 'r') as f:
        for line in f:
            tmp = line.strip().split(',')
            old_names[int(tmp[0])] = tmp[1]
            new_names[int(tmp[0])] = tmp[2]
    data = json.load(open(args.ann, 'r'))

    cat_info = copy.deepcopy(data['categories'])
    
    for x in cat_info:
        if old_names[x['id']].strip() != x['name'].strip():
            print('{} {} {}'.format(x, old_names[x['id']], new_names[x['id']]))
            import pdb; pdb.set_trace()
        if old_names[x['id']] != new_names[x['id']]:
            print('Renaming', x['id'], x['name'], new_names[x['id']])
            x['name'] = new_names[x['id']]
    
    data['categories'] = cat_info
    out_name = args.ann[:-5] + '_fixname.json'
    print('Saving to', out_name)
    json.dump(data, open(out_name, 'w'))
