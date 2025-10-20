@echo off
setlocal enabledelayedexpansion

set "folder=%cd%"
set "old_part=2510152000A"
set "new_part=2510152000A64Embed"
set "search_str=StarNet%old_part%"
set "replace_str=StarNet%new_part%"

for %%f in ("%folder%\StarNet???????????_s??.json") do (
    set "filename=%%~nxf"
    set "prefix=!filename:~0,7!"
    set "suffix=!filename:~18!"
    set "new_name=!prefix!%new_part%!suffix!"
    ren "!filename!" "!new_name!"

    set "temp_file=!new_name!.tmp"
    if exist "!temp_file!" del "!temp_file!"
    for /f "usebackq delims=" %%I in ("!new_name!") do (
        set "current_line=%%I"
        set "modified_line=!current_line:%search_str%=%replace_str%!"
        (echo !modified_line!) >> "!temp_file!"
    )

    copy /y "!temp_file!" "!new_name!" > nul
    if exist "!temp_file!" del "!temp_file!"
)