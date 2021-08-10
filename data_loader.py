import glob
from bs4 import BeautifulSoup as bs


def read_ppt_tika_xml(xml_file):
    pages_text = []
    with open(xml_file, "r") as file:
        content = "".join(file.readlines())
        bs_content = bs(content, "lxml")
        for page in bs_content.find_all("div", {"class": "page"}):
            pages_text.append([p.text.strip() for p in page.find_all("p") if p.text])
    return pages_text


def read_ppt_xml(xml_file):
    slides = []
    with open(xml_file, "r") as file:
        content = "".join(file.readlines())
        bs_content = bs(content, "lxml")
        for slide in bs_content.find_all("slide"):
            slide_text = []
            for i, bullet in enumerate(slide.findChildren(recursive=True)):
                if bullet.name == 'title':
                    slide_text.append(bullet.text)
                elif bullet.name == 'l1':
                    slide_text.append(bullet['content'])
                elif bullet.name == 'l2':
                    slide_text.append(bullet['content'])
            slides.append(slide_text)
    return slides


def read_pdf(paper):
    pdfxml = open(paper, 'rb')
    contents = pdfxml.read()
    soup = bs(contents, 'html.parser')
    abstracts = soup.find_all('abstract')
    for abstract in abstracts:
        yield [abstract.get_text()]
    divs = soup.find_all('div')
    for i, div in enumerate(divs):
        div_texts = []
        head_len = 0
        if 'type' in div.attrs and (div.attrs['type'] == 'references' or div.attrs['type'] == 'acknowledgement'):
            continue
        head = div.find('head')
        if head:
            div_texts.append(head.get_text())
        for paragraph in div.find_all('p'):
            div_texts.append(paragraph.get_text().strip())
        yield div_texts
