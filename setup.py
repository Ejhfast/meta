from distutils.core import setup
setup(
  name = 'metalang',
  packages = ['metalang'], # this must be the same as the name above
  version = '0.50',
  description = 'A domain specific language that enables powerful code sharing',
  author = 'Ethan Fast',
  author_email = 'ejhfast@gmail.com',
  url = 'https://github.com/Ejhfast/meta', # use the URL to the github repo
  download_url = 'https://github.com/Ejhfast/meta/tarball/0.50',
  keywords = ['DSL', 'code sharing', 'metaprogramming'], # arbitrary keywords
  classifiers = [],
  install_requires=[
          'numpy',
          'dill',
          'pymongo',
          'requests',
          'typing',
          'multiprocess'
  ]
)
