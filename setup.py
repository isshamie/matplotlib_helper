import atexit
from setuptools                 import setup
from setuptools.command.install import install

def _post_install():
    import goosempl
    package_name.copy_style()

class new_install(install):
    def __init__(self, *args, **kwargs):
        super(new_install, self).__init__(*args, **kwargs)
        atexit.register(_post_install)

__version__ = '0.1.0'

setup(
    name              = 'mplh',
    version           = __version__,
    install_requires  = ['matplotlib>=2.0.0','colorspacious','brewer2mpl'],
    packages          = ['mplh'],
    cmdclass          = {'install': new_install},
    package_data      = {'package_name/styles':[
        'mplh/styles/notebook_mplstyle',
    ]},
)
