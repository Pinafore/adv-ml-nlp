
clean:
	rm -r *.aux *.pdf *~ *.log *.nav *.out *.snm _minted* *.toc

01: 01a.pdf 01b.pdf 01c.pdf 01d.pdf 01e.pdf 01f.pdf

02: 02a.pdf 02b.pdf 02c.pdf 02d.pdf

03: 03a.pdf 03b.pdf 03c.pdf 03d.pdf

04: 04a.pdf 04b.pdf

05: 05a.pdf 05b.pdf 05c.pdf 05d.pdf

06: 06a.pdf 06b.pdf 06c.pdf 06d.pdf

07: 07a.pdf 07b.pdf 07c.pdf

09: 09a.pdf 09b.pdf 09c.pdf

10: 10a.pdf 10b.pdf 10c.pdf

11: 11a.pdf 11b.pdf 11c.pdf 11d.pdf

12: 12a.pdf 12b.pdf 12c.pdf 12d.pdf 12e.pdf

%.pdf: %.tex
	pdflatex -shell-escape $*
	cp $@ ../../jbg-web/teaching/CSCI_7000
