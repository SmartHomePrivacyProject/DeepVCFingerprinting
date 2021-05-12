# Speaker Data Collection Platform
Automate Smart Speaker Data Collection.  

<img src=echo.jpg />.

This platform allows for automatic generation and collection of network traffic data from smart assistant-enabled speakers like the Amazon Echo and the Google Home.  Audio files are generated corresponding to queries (like 'Alexa, what is the weather today?').  These files are played from a regular speaker initiating an interaction between the speaker and the smart speaker.  At the same time a network sniffer is started on the AP the smart speaker is connected to.  The network traffic associated with the interaction is saved to a pcap file.  

## Setup

In order to set up a speaker data collection platform the following items are required:

- Smart Speaker (Amazon Echo or Google Home)
- Normal Speaker
- Raspberry Pi setup as Wireless Access Point

### Setting up the Raspberry Pi
The Raspberry Pi needs to be configured as a wireless access point. This setup is assuming that the Raspberry Pi is connected to the router via ethernet cable.

On a Raspberry Pi running Raspbian OS, install `bridge-utils`, `dnsmasq` and `hostapd`:

```
sudo apt install hostapd
sudo apt install dnsmasq
sydo apt install bridge-utils
```

Then, stop the processes with `systemd`:

```
sudo systemctl stop hostapd
sudo systemctl stop dnsmasq
```

A static IP address must be assigned to the wireless interface on the Raspberry Pi. Use `iwconfig` to discover the name of the wireless interface. Edit the `/etc/dhcpcd.conf` file and append this text to the end, replacing `wlan0` with the name of your interface:

```
interface wlan0
static ip_address=192.168.0.10/24
denyinterfaces eth0
denyinterfaces wlan0
```

`dnsmasq` will function as a DHCP sever. Edit `/etc/dnsmasq.conf` to this:

```
interface=wlan0
  dhcp-range=192.168.0.11,192.168.0.30,255.255.255.0,24h
```

Now, edit the `/etc/hostapd/hostapd.conf` file and add the following, substituting whichever SSID and password you like:

```
interface=wlan0
bridge=br0
hw_mode=g
channel=7
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
ssid=SSID_NAME
wpa_passphrase=PASSWORD123
```

Once that's done, edit the `/etc/default/hostapd` file and uncomment and change the following line:

```
DAEMON_CONF="/etc/hostapd/hostapd.conf"
```

Edit the `/etc/sysctl.conf` file and uncomment `net.ipv4.ip_forward=1`. Now run the following command to setup `iptables`:

```
sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
sudo sh -c "iptables-save > /etc/iptables.ipv4.nat"
```

Edit the `/etc/rc.local` file and add this line above `exit 0`:

```
iptables-restore < /etc/iptables.ipv4.nat
```

Add the bridge and connect the `eth0` interface to the bridge:

```
sudo brctl addbr br0
sudo brctl addif br0 eth0
```

Lastly, edit the `/etc/network/interfaces` file and add the following to the end:

```
auto br0
iface br0 inet manual
bridge_ports eth0 wlan0
```

Test the newly created access point by attempting to connect to it with a smartphone or other device.


### Setting up the Smart Speaker
Hold down the action button on the Amazon Echo until an orange ring appears to enter setup mode. With the Alex app, connect the Amazon Echo to the newly created wireless access point with the credentials you created in the `/etc/hostapd/hostapd.conf` file.

Now, place connect the normal speaker to the Raspberry Pi and ensure the speakers are close enough and that the volume is adequate to trigger the smart speaker. 


## Usage

In addition to the physical setup described above, you need to have a csv file containing the interaction time for each query (i.e., the query 'Alexa, what is the weather today?' takes 30 seconds). The `csv` file should be in the form `request, time`.

### Generating Audio Files 

Use the generate_audio_queries.py script to produce mp3 files that contain the queries the speaker will play.  By default the wake work used is "Alexa, " and the voice used is the default Google text-to-speech US-English Female voice. The `--wake_word` argument can be used to specify the wake-up word and `--voice` can be used to specify which voice can be used (either `joanna`, `salli`, `matt`, `ivy`, `kevin`, `justin`, `kimberly`, `kendra` or `joey`). Note that these AWS Polly voices require valid AWS  credentials. To setup the credentials, ensure that `~/.aws/credentials` contains:

```
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
```
Also, ensure `~/.aws/config` contains a valid region (tested with `eu-west-2`):

```
[default]
region=eu-west-2
```

```bash
./generate_audio_queries.py [--voice voice] [--wake-word word] data/queries.csv 
```

### Capturing Network Traffic Data of Smart Speaker Interactions

Use the speaker_collect.py script to start the platform. You need to specify the CSV File that contains the columns Query and Time, corresponding to the query and interaction time, the  directory containing the audio query files, the device's IP address, and the number of captures to collect for each query.

```bash
sudo ./speaker_collect.py data/file.csv voice_queries/ 192.192.192.192 100
```



