# Test code for AttnGAN model image generation

from model.UI import Text_to_Image, RandomCaption

if __name__ == "__main__":
    t2i = Text_to_Image()
    with open('result.html', 'w') as f:
        f.write('''
        <html>
        <body>
            <table border='1px'>
        ''')

        for i in range(1):
            text = RandomCaption().get()
            im = t2i.process(text)
            data = '</td>\n\t\t\t\t</tr>\n\t\t\t\t<tr>\n\t\t\t\t\t<td>'.join(['</td>\n\t\t\t\t\t<td>'.join([str(k) for k in j]) for j in im[2]])
            f.write('''
            <tr>
                <td> <img src="{}"></img> </td>
                <td> <b> {} </b>
				<br><br>
				<table>
				<tr>
					<td> {} </td>
				</tr>
				</table> </td>
            </tr>
            '''.format(im[3], text, data))
        
        f.write('''
        </table>
        </body>
        </html>
        ''')