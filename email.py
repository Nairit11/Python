import smtplib as sm
smtpobj=sm.SMTP('smtp.gmail.com',587)
smtpobj.ehlo()
smtpobj.starttls()
smtpobj.login('YOUR_EMAIL_ID','YOUR_PASSWORD')
smtpobj.sendmail('YOUR_EMAIL_ID','RECEIVER_EMAIL_ID','Subject: Python test.\n This code is to test the smtp library\n to send emails.\n\n')
smtpobj.quit()