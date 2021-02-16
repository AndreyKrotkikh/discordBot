import discord


class MyClient(discord.Client):
    async def on_ready(self):
        print('Logged on as {0}!'.format(self.user))
        for guild in self.guilds:
            print('Logged on {0}'.format(guild.name))

    async def on_message(self, message):
        #print(message.author.name)
        #print(message.author.id)
        if message.author.id == 366921098077667338 and message.content.startswith('++'):
            channel = message.channel
            await channel.send(content=message.content)
        print('Message from {0.author} on {0.guild} on {0.channel}: {0.content}'.format(message))


client = MyClient()
client.run('NTg5NDQxOTgzNTA5MTY4MTI4.XQTuoA.n8SbAoYYnb4YX0OmKfQcwJyZG2U')
