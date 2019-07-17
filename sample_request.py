import requests

url = 'http://127.0.0.1:1996/sentiment'

r = requests.post(url,json={"reviews":["Instead, I went once again to the 24/7 Cafe.  I ordered their prime rib special that comes with vegetable and either mashed or loaded baked potato.  I ordered the beef extra rare and that's the way it came.  Sure, this place isn't Lawry's but the meat was excellent and the potato and broccoli were very good.","The prices were about $2 more than in Chinatown, and although it is closer to our house, the pho broth does not cut it. The broth is underflavored and a bit too sweet. The noodles in the XL bowl are not enough and the spring rolls lacked meat and noodles. The service was not friendly, and the place was so quiet, it was eerie. What is it about the Palms buffet that there's always a line when I am going there.  So many people complain about it, you'd think I could walk right in.","For dessert, I got a piece of chocolate cake.  You can't ever pass that up."], "sentiment":[1,0,1]})

print(r.json())