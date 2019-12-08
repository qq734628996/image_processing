@echo off
chcp 65001
cls

set ARTICLE=report
xelatex -synctex=1 %ARTICLE%
xelatex -synctex=1 %ARTICLE%
call clean
