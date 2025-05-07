from kan.MultKAN import MultKAN as KAN

class KANWrapper(KAN):
    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        return None, output  