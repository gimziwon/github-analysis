#!/usr/bin/python
import sys, os, json
import fnmatch

path = '/mnt/NAS/hgb/githubraw/'

out_file = '/mnt/NAS/georgewang/data.' + sys.argv[1]
err_file = '/mnt/NAS/georgewang/data.' + sys.argv[1] + '.error'

wp = open(out_file, 'w')
ep = open(err_file, 'w')

for file in os.listdir(path):
    if fnmatch.fnmatch(file, '{}-*.json'.format(sys.argv[1])):
        print('Run {}...'.format(file), file=sys.stderr)

        with open(os.path.join(path, file), 'r') as reader:
            lines = reader.readlines()

        for line in lines:
            try:
                obj = json.loads(line.strip())
            except:
                print(line)

            try:
                record_id = obj['id']
                actor_name = obj['actor']['login']
                repo_name = obj['repo']['name']
                event_name = obj['type']

                wp.write('{}\t{}\t{}\t{}\n'.format(record_id, event_name, actor_name, repo_name))
            except:
                ep.write('{}\n'.format(json.dumps(obj)))

wp.close()
ep.close()
