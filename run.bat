@echo off
REM 设置C++程序的位置
set "CPP_PROGRAM=%cd%\build\Release\HGRSTAR.exe"

REM 设置包含JSON文件的目录
set "JSON_DIR=%cd%\json"

REM 进入JSON文件所在的目录
cd /d %JSON_DIR%

echo "%CPP_PROGRAM%"

REM 遍历当前目录下的所有JSON文件，并调用C++程序处理每个文件
for %%f in (*.json) do (
    echo Processing file: %%f
    "%CPP_PROGRAM%" "%%f"
)

echo All files have been processed.
pause