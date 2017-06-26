# This makefile does nothing but delegating the actual building to cmake.

# $(shell python ./scripts/get_python_cmake_flags.py) 输出的是 Python2.7 的头文件目录
# 我的 Ubuntu 输出为 -DPYTHON_INCLUDE_DIR=/home/long/anaconda2/include/python2.7
all:
	@mkdir -p build && cd build && cmake .. $(shell python ./scripts/get_python_cmake_flags.py) && $(MAKE)

local:
	@./scripts/build_local.sh

android:
	@./scripts/build_android.sh

ios:
	@./scripts/build_ios.sh

clean: # This will remove ALL build folders.
	@rm -r build*/

linecount:
	@cloc --read-lang-def=caffe.cloc caffe2 || \
		echo "Cloc is not available on the machine. You can install cloc with " && \
		echo "    sudo apt-get install cloc"
