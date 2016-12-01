
clean:
	rm -r *.aux *.pdf *~ *.log *.nav *.out *.snm _minted* *.toc

01: 01a.pdf 01b.pdf

%.pdf: %.tex
	pdflatex -shell-escape $*
	cp $@ ../../../jbg-web/teaching/CSCI_3022
