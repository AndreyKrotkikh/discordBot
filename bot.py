import discord


class MyClient(discord.Client):
    async def on_ready(self):
        print('Logged on as {0}!'.format(self.user))
        for guild in self.guilds:
            print('Logged on {0}'.format(guild.name))

    async def on_message(self, message):
        if message.author == 'Giallar#0623':
            discord.Message.to_reference(message)
        print('Message from {0.author} on {0.guild} on {0.channel}: {0.content}'.format(message))


client = MyClient()
client.run('NTg5NDQxOTgzNTA5MTY4MTI4.XQTuoA.n8SbAoYYnb4YX0OmKfQcwJyZG2U')
