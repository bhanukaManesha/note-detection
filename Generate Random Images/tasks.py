#!/usr/bin/env python3
import glob
from fabric import Connection
from invoke import task

HOST        = 'ec2-52-221-236-106.ap-southeast-1.compute.amazonaws.com'
USER        = 'ubuntu'
ROOT        = 'generator'
REMOTE      = '{user}@{host}:{root}'.format(user=USER, host=HOST, root=ROOT)
VENV        = 'virtualenv'
LOCAL_FILES = [
    'generate.py',
    'background.tar',
    'images.tar'
]

IMAGES = [
    'data'
]



@task
def connect(ctx):
    ctx.conn = Connection(host=HOST, user=USER)

@task
def close(ctx):
    ctx.conn.close()

@task(pre=[connect], post=[close])
def setup(ctx):
    ctx.conn.run('mkdir -p {}'.format(ROOT))
    with ctx.conn.cd(ROOT):
        ctx.conn.run('sudo apt install -y dtach')
    # PIP
    ctx.conn.put('requirements.in', remote='{}/requirements.in'.format(ROOT))
    with ctx.conn.cd(ROOT):
        with ctx.conn.prefix('source activate virtualenv'):
            ctx.conn.run('pip install -U pip')
            ctx.conn.run('pip install --upgrade pip')
            ctx.conn.run('pip install pip-tools')
            ctx.conn.run('pip-compile --upgrade requirements.in')
            ctx.conn.run('pip-sync')

@task
def push(ctx, model=''):
    ctx.run('rsync -rv --progress {files} {remote}'.format(files=' '.join(LOCAL_FILES), remote=REMOTE))
    model = sorted([fp for fp in glob.glob('models/*') if model and model in fp], reverse=True)

@task
def pull(ctx):
    ctx.run('rsync -r {remote}/{folder}/ {folder}'.format(remote=REMOTE, folder=MODEL))
    ctx.run('rsync -r {remote}/{folder}/ {folder}'.format(remote=REMOTE, folder=OUTPUT))

@task(pre=[connect], post=[close])
def train(ctx):
    with ctx.conn.cd(ROOT):
        with ctx.conn.prefix('source activate tensorflow2_p36'):
            ctx.conn.run('dtach -A /tmp/{} python train.py'.format(ROOT), pty=True)

@task(pre=[connect], post=[close])
def resume(ctx):
    ctx.conn.run('dtach -a /tmp/{}'.format(ROOT), pty=True)

@task(pre=[connect], post=[close])
def test(ctx, model=''):
    with ctx.conn.cd(ROOT):
        with ctx.conn.prefix('source activate tensorflow2_p36'):
            ctx.conn.run('python test.py {}'.format(model), pty=True)

@task(pre=[connect], post=[close])
def clean(ctx):
    with ctx.conn.cd(ROOT):
        ctx.conn.run('rm -rf {}/*'.format(MODEL), pty=True)
