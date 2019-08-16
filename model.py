import torch
import torch.nn as nn
import torch.nn.functional as F
from config import opt


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		m.bias.data.fill_(0)

		nn.init.xavier_uniform_(m.weight,gain=0.5)


class encoder_cada(nn.Module):
	"""
	This is the encoder class which consists of the
	encoder for features and the attributes.

	features: x
	attributes: att
	"""
	def __init__(self, input_dim=2048, atts_dim=312, z=64 ):
		super(encoder_cada, self).__init__()
		self.encoder_x = nn.Sequential(nn.Linear(input_dim, 1560), nn.ReLU())
		self.mu_x = nn.Linear(1560, z)
		self.logvar_x = nn.Linear(1560, z)		
		
		self.encoder_att = nn.Sequential(nn.Linear(atts_dim, 1450), nn.ReLU())
		self.mu_att = nn.Linear(1450, z)
		self.logvar_att = nn.Linear(1450, z)

		self.apply(weights_init)

	def reparameterize(self, mu, logvar):
		sigma = torch.exp(logvar)
		eps = torch.FloatTensor(logvar.size()[0], 1).normal_(0, 1).to(opt.device)  # 50*1  从单位高斯随机采样
		eps = eps.expand(sigma.size())  # 50*64
		return mu + sigma*eps  # 通过潜分布的均值改变分布的均值，通过方差对其进行缩放

	def forward(self, x, att):
		x = self.encoder_x(x)
		mu_x = self.mu_x(x)
		logvar_x = self.logvar_x(x)
		z_x = self.reparameterize(mu_x, logvar_x)

		att = self.encoder_att(att)
		mu_att = self.mu_att(att)
		logvar_att = self.logvar_att(att)
		z_att = self.reparameterize(mu_att,logvar_att)
		return z_x, z_att, mu_x, logvar_x, mu_att, logvar_att


class decoder_cada(nn.Module):
	"""docstring for decoder_cada"""
	def __init__(self, input_dim=2048, atts_dim=312, z=64):
		super(decoder_cada, self).__init__()
		self.decoder_x = nn.Sequential(nn.Linear(z, 1660), nn.ReLU(), nn.Linear(1660, input_dim))
		self.decoder_att = nn.Sequential(nn.Linear(z, 665),nn.ReLU(), nn.Linear(665, atts_dim))

		self.apply(weights_init)


	def forward(self, z_x, z_att):
		recon_x = self.decoder_x(z_x)
		recon_att = self.decoder_att(z_att)
		att_recon_x = self.decoder_att(z_x)
		x_recon_att = self.decoder_x(z_att)
		return recon_x, recon_att, att_recon_x, x_recon_att


class Classifier(nn.Module):
	def __init__(self, input_dim, num_class):
		super(Classifier, self).__init__()
		self.fc = nn.Linear(input_dim, num_class)  # 64*50
		self.softmax = nn.LogSoftmax(dim=1)

		self.apply(weights_init)

	def forward(self, features):
		x = self.softmax(self.fc(features))
		return x


class CADA_VAE(nn.Module):
	def __init__(self, input_dim=2048, atts_dim=312, z=64):
		super(CADA_VAE, self).__init__()
		self.fc1_x = nn.Linear(input_dim, 1560)
		self.fc21_x = nn.Linear(1560, z)
		self.fc22_x = nn.Linear(1560, z)
		self.fc3_x = nn.Linear(z, 1660)
		self.fc4_x = nn.Linear(1660, input_dim)

		self.fc1_sig = nn.Linear(atts_dim, 1450)
		self.fc21_sig = nn.Linear(1450, z)
		self.fc22_sig = nn.Linear(1450, z)
		self.fc3_sig = nn.Linear(z, 660)
		self.fc4_sig = nn.Linear(660, atts_dim)

	def encode_x(self, x):
		h1 = F.relu(self.fc1_x(x))
		return self.fc21_x(h1), self.fc22_x(h1)

	def decode_x(self, z):
		h3 = F.relu(self.fc3_x(z))
		return torch.sigmoid(self.fc4_x(h3))

	def encode_sig(self, sig):
		h1 = F.relu(self.fc1_sig(sig))
		return self.fc21_sig(h1), self.fc22_sig(h1)

	def decode_sig(self, z):
		h3 = F.relu(self.fc3_sig(z))
		return torch.sigmoid(self.fc4_sig(h3))

	def reparameterize(self, mu, logvar):
		std = torch.exp(logvar)
		eps = torch.randn_like(std)  # mean 0, std  输入数据求尺寸input.size()，调用torch.randn标准正太分布
		return eps.mul(std).add_(mu)  # 标量乘法

	def forward(self, x, sig):
		# VAE for feature vector
		mu_x, logvar_x = self.encode_x(x)  # 返回均值和方差
		z_x = self.reparameterize(mu_x, logvar_x)  # 再参数化
		recon_x = self.decode_x(z_x)
		#         import pdb;pdb.set_trace()

		# VAE for signature vector
		mu_sig, logvar_sig = self.encode_sig(sig)
		z_sig = self.reparameterize(mu_sig, logvar_sig)
		recon_sig = self.decode_sig(z_sig)

		# CA
		sigDecoder_x = self.decode_sig(z_x)
		xDecoder_sig = self.decode_x(z_sig)

		return recon_x, recon_sig, mu_x, mu_sig, sigDecoder_x, xDecoder_sig, logvar_x, logvar_sig


