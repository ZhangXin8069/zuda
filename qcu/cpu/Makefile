TARGET:=libqcu.so qcu_cpu.o
LIB:=libqcu.so
INCLUDE:=${HOME}/external-libraries/pyquda/include/qcu
OBJECTS:=qcu_cpu.o
SRC:=qcu_cpu.cc
FLAGS:= -O3 # -g
INSTALL_DIR:=../../lib

${LIB}: ${OBJECTS}
	g++ -shared -o ${LIB} ${OBJECTS}

${OBJECTS}: ${SRC} # qcu.h
	g++ -c ${SRC}  ${FLAGS} -I ${INCLUDE} -fPIC

.PHONY: install clean
install:
	cp ${TARGET} ${INSTALL_DIR}/
clean:
	-rm ${TARGET}
