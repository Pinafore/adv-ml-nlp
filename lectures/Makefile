
clean:
	rm -r *.aux *.pdf *~ *.log *.nav *.out *.snm _minted* *.toc

01: 01a.pdf 01b.pdf 01c.pdf 01d.pdf 01e.pdf

%.pdf: %.tex
	pdflatex -shell-escape $*
	cp $@ ../../jbg-web/teaching/CSCI_7000
