#!/usr/bin/python
import sys, os, json
import fnmatch

path = '/mnt/NAS/hgb/githubraw/'

out_file = '/mnt/disk2/georgewang/data.' + sys.argv[1] + '.lang'
err_file = '/mnt/disk2/georgewang/data.' + sys.argv[1] + '.lang.error'

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
                ep.write('{}\n'.format(line))

            try:
                if obj['payload']['pull_request']['head']['repo']['language'] is not None:
                    lang_name = obj['payload']['pull_request']['head']['repo']['language']
                    repo_name = obj['repo']['name']
                    wp.write('{}\t{}\n'.format(repo_name, lang_name))
            except:
                ep.write('{}\n'.format(json.dumps(obj)))

wp.close()
ep.close()
