# 指定 node 版本
FROM bitnami/python:3.11 as dev

# 设置编码
ENV LANG C.UTF-8

# 设置工作目录
WORKDIR /root/FutuTradingAI

ENV SHELL /bin/bash

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN if command -v apt-get &> /dev/null; then \
        sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
        apt-get update && apt-get update && apt-get install -y sudo curl git zsh tree vim; \
        sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"; \
        sh -c 'git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting'; \
        sh -c 'git clone https://github.com/zsh-users/zsh-autosuggestions.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions'; \
        echo 'export ZSH=$HOME/.oh-my-zsh' >> /root/.zshrc; \
        echo 'ZSH_THEME="ys"' >> /root/.zshrc; \
        echo 'plugins=(git zsh-syntax-highlighting zsh-autosuggestions)' >> /root/.zshrc; \
        echo 'source $ZSH/oh-my-zsh.sh' >> /root/.zshrc; \
   else \
        echo "apt-get command does not exist."; \
    fi

# 安装 docker-cli 和 docker-compose-plugin（不含 docker engine）
RUN if command -v apt-get &> /dev/null; then \
        apt-get update && \
        apt-get install -y ca-certificates curl gnupg && \
        install -m 0755 -d /etc/apt/keyrings && \
        curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
        chmod a+r /etc/apt/keyrings/docker.gpg && \
        echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://mirrors.aliyun.com/docker-ce/linux/debian \
        "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null && \
        apt-get update && \
        apt-get install -y docker-ce-cli docker-compose-plugin; \
    else \
        echo "apt-get command does not exist."; \
    fi

# 设置 zsh 为默认 shell
ENV SHELL /bin/zsh
